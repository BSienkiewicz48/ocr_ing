# app.py
# -*- coding: utf-8 -*-

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import streamlit as st


# =============================
# USTAWIENIA / HASŁO (PLAINTEXT)
# =============================
# .streamlit/secrets.toml
#  wariant A:
#    password = "TwojeHaslo"
#  wariant B:
#    [auth]
#    password = "TwojeHaslo"
PASSWORD = (
    st.secrets.get("password")
    or (st.secrets.get("auth", {}) or {}).get("password")
)

# -----------------------------
# FUNKCJE POMOCNICZE
# -----------------------------
def check_password() -> bool:
    """Prosty gate hasłem (plaintext) zapisany w st.secrets."""
    if PASSWORD is None:
        st.error(
            "Brak hasła w `st.secrets`. Dodaj `password = \"...\"` "
            "lub `[auth]\npassword = \"...\"` w `.streamlit/secrets.toml`."
        )
        return False

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    with st.form("login", clear_on_submit=False):
        pwd = st.text_input("Hasło", type="password")
        submitted = st.form_submit_button("Zaloguj")

    if submitted:
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            return True
        else:
            st.error("Błędne hasło.")
            return False
    else:
        st.info("Podaj hasło, aby skorzystać z aplikacji.")
        return False


def extract_df_from_pdf(file_bytes: bytes) -> pd.DataFrame:
    """
    Otwiera PDF z bajtów i zwraca surowy DataFrame 'df_OCR' z polami:
    Strona, Tekst, X0, Y0, X1, Y1
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    data = []

    for page_number, page in enumerate(doc, start=1):
        text_instances = page.get_text("dict")["blocks"]
        for instance in text_instances:
            if instance.get("type", None) == 0:
                for line in instance.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox", [None, None, None, None])
                        data.append(
                            {
                                "Strona": page_number,
                                "Tekst": span.get("text", ""),
                                "X0": float(bbox[0]) if bbox[0] is not None else np.nan,
                                "Y0": float(bbox[1]) if bbox[1] is not None else np.nan,
                                "X1": float(bbox[2]) if bbox[2] is not None else np.nan,
                                "Y1": float(bbox[3]) if bbox[3] is not None else np.nan,
                            }
                        )
    doc.close()
    df = pd.DataFrame(data)
    return df


def process_dataframe(df_OCR: pd.DataFrame) -> pd.DataFrame:
    """
    PRZETWARZANIE IDENTYCZNE jak w dostarczonym skrypcie:
    - filtry Y0, tekstowe, X0
    - grupowanie po zmianach X0 (bez zaokrągleń)
    - mapowanie Text (z X0==246 po X0==81), Amount (pierwsze X0 w [418,450.5] po X0==81)
    - Date (z X0 w [30.19,30.20] do najbliższego kolejnego X0==81)
    - kolumny końcowe: Page, Date, Partner name, Text, Amount, Currency, FV
    - FV: czyszczenie, PL/DE prefiks, regex (PL|DE)\d{10}, unikalizacja
    """

    # --- Filtry na podstawie pozycji Y0 (usuwamy nagłówki/stopki) ---
    df_OCR = df_OCR.loc[~df_OCR["Y0"].between(29.32, 80)]
    df_OCR = df_OCR.loc[~df_OCR["Y0"].between(765.52, 765.53)]
    # w oryginale porównanie było do '1' (string); rzutujemy 'Strona' na str
    df_OCR = df_OCR.loc[~((df_OCR["Y0"].between(29.32, 278)) & (df_OCR["Strona"].astype(str) == "1"))]

    # --- Filtry tekstowe ---
    df_OCR = df_OCR[~df_OCR["Tekst"].str.startswith("Wygenerowano", na=False)]
    df_OCR = df_OCR[~df_OCR["Tekst"].isin(
        [
            "Kontrahent",
            "Tytuł operacji",
            "Typ operacji",
            "Kwota",
            "Waluta",
            "Saldo po operacji",
            "Rachunek firmy",
            "uznanie",
        ]
    )]

    # Usuwanie wierszy, które zawierają wzór numeru (np. numer rachunku)
    pattern = r"\b\d{2}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\b"
    df_OCR = df_OCR[~df_OCR["Tekst"].str.contains(pattern, na=False, regex=True)]

    # --- Filtry po X0 ---
    # (dokładnie te same zakresy co w skrypcie docelowym)
    df_OCR = df_OCR.loc[~df_OCR["X0"].between(519.141, 519.142)]
    df_OCR = df_OCR.loc[~df_OCR["X0"].between(500, 530)]
    df_OCR = df_OCR.loc[~df_OCR["X0"].between(263, 264)]

    # --- Grupowanie po zmianach X0 (BEZ zaokrąglania; jak w skrypcie) ---
    df_OCR = df_OCR.sort_values(by=["Strona", "Y0", "X0"]).reset_index(drop=True)
    df_OCR["group"] = (df_OCR["X0"] != df_OCR["X0"].shift()).cumsum()
    df_OCR["Tekst"] = df_OCR.groupby(["X0", "group"])["Tekst"].transform(" ".join)
    df_OCR = df_OCR.drop_duplicates(subset=["X0", "group"]).drop(columns="group")

    # --- Inicjalizacja kolumn ---
    df_OCR["Text"] = None
    df_OCR["Amount"] = None

    # --- Mapowanie: DOKŁADNE porównania X0 (==) jak w skrypcie ---
    for index in df_OCR[df_OCR["X0"] == 81].index:
        next_246_index = df_OCR.loc[index + 1 :, "X0"][df_OCR["X0"] == 246].index.min()
        if pd.notna(next_246_index):
            df_OCR.loc[index, "Text"] = df_OCR.loc[next_246_index, "Tekst"]

        next_amount_index = (
            df_OCR.loc[index + 1 :, "X0"]
            .where(df_OCR["X0"].between(418, 450.5))
            .dropna()
            .index.min()
        )
        if pd.notna(next_amount_index):
            df_OCR.loc[index, "Amount"] = df_OCR.loc[next_amount_index, "Tekst"]

    # --- Kolumna Date: X0 w [30.19, 30.20] -> do najbliższego następnego X0 == 81 ---
    df_OCR["Date"] = None
    for index in df_OCR[(df_OCR["X0"] >= 30.19) & (df_OCR["X0"] <= 30.20)].index:
        next_81_index = df_OCR.loc[index + 1 :, "X0"][df_OCR["X0"] == 81].index.min()
        if pd.notna(next_81_index):
            df_OCR.loc[next_81_index, "Date"] = df_OCR.loc[index, "Tekst"]

    # --- Porządki kolumn (jak w skrypcie docelowym) ---
    df_OCR.drop(["X0", "Y0", "X1", "Y1"], axis=1, inplace=True, errors="ignore")
    df_OCR.dropna(subset=["Text"], inplace=True)

    df_OCR["Partner name"] = df_OCR["Tekst"]
    df_OCR.drop(columns="Tekst", inplace=True, errors="ignore")
    df_OCR.rename(columns={"Strona": "Page"}, inplace=True)

    # Kolejność kolumn
    df_OCR = df_OCR[["Page", "Date", "Partner name", "Text", "Amount"]]

    # --- Rozbicie Amount na Currency i Amount (identycznie: ostatnie 3 znaki) ---
    df_OCR["Currency"] = df_OCR["Amount"].astype(str).str[-3:]
    df_OCR["Amount"] = df_OCR["Amount"].astype(str).str[:-3]

    # --- FV jak w skrypcie docelowym ---
    df_OCR["FV"] = df_OCR["Text"].copy()
    df_OCR["FV"] = df_OCR["FV"].str.replace(" ", "", regex=True)
    df_OCR["FV"] = df_OCR["FV"].str.upper()

    # Dodawanie przedrostków PL/DE (tylko jeśli brak istniejącego prefiksu)
    df_OCR["FV"] = df_OCR["FV"].apply(
        lambda x: x.replace("24270", "PL24270") if ("24270" in x and "PL" not in x) else x
    )
    df_OCR["FV"] = df_OCR["FV"].apply(
        lambda x: x.replace("24280", "DE24280") if ("24280" in x and "DE" not in x) else x
    )

    # Ekstrakcja kodów PL/DE + 10 cyfr
    def extract_codes(value: str):
        if not isinstance(value, str):
            return None
        matches = re.findall(r"(?:PL|DE)\d{10}", value)
        return " ".join(matches) if matches else None

    df_OCR["FV"] = df_OCR["FV"].apply(extract_codes)

    # Usunięcie duplikatów kodów w obrębie jednego wiersza
    def remove_duplicates(val):
        if pd.isna(val):
            return val
        values = str(val).split()
        unique_values = list(dict.fromkeys(values))
        return " ".join(unique_values)

    df_OCR["FV"] = df_OCR["FV"].apply(remove_duplicates)

    # Ostateczna kolejność kolumn
    df_OCR = df_OCR[["Page", "Date", "Partner name", "Text", "Amount", "Currency", "FV"]]

    return df_OCR


def split_column_to_rows(df: pd.DataFrame, column_to_split: str) -> pd.DataFrame:
    """
    Rozbija wartości z kolumny (spacja-separowane) na osobne wiersze,
    kopiując pozostałe kolumny. (logika jak w skrypcie docelowym)
    """
    rows = []
    for _, row in df.iterrows():
        if pd.notna(row.get(column_to_split)):
            split_values = str(row[column_to_split]).split()
            for value in split_values:
                new_row = row.to_dict()
                new_row[column_to_split] = value
                rows.append(new_row)
        else:
            new_row = row.to_dict()
            new_row[column_to_split] = None
            rows.append(new_row)
    return pd.DataFrame(rows)


def to_excel_bytes(df: pd.DataFrame, sheet_name="Payments_OCR") -> bytes:
    """Zapisuje DataFrame do Excela (w pamięci) i zwraca bajty."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


# -----------------------------
# UI STREAMLIT
# -----------------------------
st.set_page_config(page_title="PDF → Excel (OCR wyciąg płatności)", page_icon="📄", layout="wide")

st.title("📄➡️📊 PDF → Excel: wyciąg płatności (OCR)")
st.caption("PyMuPDF + pandas | logika OCR zgodna z Twoim skryptem referencyjnym (ING)")

# Gate hasłem
if not check_password():
    st.stop()

st.success("Dostęp przyznany.")

with st.expander("Instrukcja", expanded=False):
    st.markdown(
        """
        1. Wgraj plik **PDF** z wyciągiem.  
        2. Kliknij **Przetwórz**.  
        3. Pobierz **Excel** z wynikami lub obejrzyj podgląd tabeli.  

        > Uwaga: logika przetwarzania odpowiada dokładnie podanemu skryptowi (ING).
        """
    )

uploaded = st.file_uploader("Wgraj plik PDF", type=["pdf"])

col_a, col_b = st.columns([1, 2])
with col_a:
    process_btn = st.button("🚀 Przetwórz", type="primary", use_container_width=True)

if uploaded and process_btn:
    try:
        pdf_bytes = uploaded.read()
        raw_df = extract_df_from_pdf(pdf_bytes)
        if raw_df.empty:
            st.warning("Nie znaleziono tekstu w PDF.")
        else:
            df_OCR = process_dataframe(raw_df)
            new_df = split_column_to_rows(df_OCR, "FV")

            # Nazwa pliku wyjściowego na podstawie nazwy wejściowej
            base_filename = uploaded.name.rsplit(".", 1)[0]
            excel_bytes = to_excel_bytes(new_df, sheet_name="Payments_OCR")

            st.success("Przetwarzanie zakończone.")
            st.download_button(
                label="📥 Pobierz Excel",
                data=excel_bytes,
                file_name=f"{base_filename}_Payments_OCR_new.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            with st.expander("Podgląd wyników (pierwsze wiersze)"):
                st.dataframe(new_df.head(200), use_container_width=True)

    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

elif not uploaded and process_btn:
    st.warning("Najpierw wgraj plik PDF.")

st.divider()

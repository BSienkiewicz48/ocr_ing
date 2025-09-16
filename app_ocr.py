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
    (kolejność rekordów = kolejność zwrócona przez PyMuPDF)
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
    return pd.DataFrame(data)


def process_dataframe(df_OCR: pd.DataFrame) -> pd.DataFrame:
    """
    PRZETWARZANIE 1:1 z Twoim skryptem bazowym (bez sortowania przed grupowaniem):
    - filtry Y0, tekstowe, X0
    - group = (X0 != shift(X0)).cumsum(), Tekst = transform(' '.join) per (X0, group)
    - Text: dla X0==81 -> pierwszy kolejny X0==246; Amount: pierwszy kolejny X0∈[418,450.5]
    - Date: wiersz z X0∈[30.19,30.20] -> najbliższy następny X0==81
    - końcowe kolumny + FV jak w bazie
    """

    # --- Filtry Y0 (nagłówki/stopki) ---
    df_OCR = df_OCR.loc[~df_OCR["Y0"].between(29.32, 80)]
    df_OCR = df_OCR.loc[~df_OCR["Y0"].between(765.52, 765.53)]
    df_OCR = df_OCR.loc[
        ~((df_OCR["Y0"].between(29.32, 278)) & (df_OCR["Strona"].astype(str) == "1"))
    ]

    # --- Filtry tekstowe ---
    df_OCR = df_OCR[~df_OCR["Tekst"].str.startswith("Wygenerowano", na=False)]
    df_OCR = df_OCR[
        ~df_OCR["Tekst"].isin(
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
        )
    ]

    # Usuwanie wzoru liczbowego (np. numer rachunku)
    pattern = r"\b\d{2}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\b"
    df_OCR = df_OCR[~df_OCR["Tekst"].str.contains(pattern, regex=True, na=False)]

    # --- Filtry X0 ---
    df_OCR = df_OCR.loc[~df_OCR["X0"].between(519.141, 519.142)]
    df_OCR = df_OCR.loc[~df_OCR["X0"].between(500, 530)]
    df_OCR = df_OCR.loc[~df_OCR["X0"].between(263, 264)]

    # >>> KLUCZOWE: BEZ SORTOWANIA <<<
    # Grupowanie po zmianach X0 (dokładnie jak w bazie)
    df_OCR["group"] = (df_OCR["X0"] != df_OCR["X0"].shift()).cumsum()
    df_OCR["Tekst"] = df_OCR.groupby(["X0", "group"])["Tekst"].transform(" ".join)
    df_OCR = df_OCR.drop_duplicates(subset=["X0", "group"]).drop(columns="group")

    # Inicjalizacja
    df_OCR["Text"] = None
    df_OCR["Amount"] = None

    # Mapowanie: X0==81 -> Text z pierwszego kolejnego X0==246, Amount z [418,450.5]
    for idx in df_OCR[df_OCR["X0"] == 81].index:
        next_246 = df_OCR.loc[idx + 1 :, "X0"][df_OCR["X0"] == 246].index.min()
        if pd.notna(next_246):
            df_OCR.loc[idx, "Text"] = df_OCR.loc[next_246, "Tekst"]

        next_amt = (
            df_OCR.loc[idx + 1 :, "X0"].where(df_OCR["X0"].between(418, 450.5)).dropna().index.min()
        )
        if pd.notna(next_amt):
            df_OCR.loc[idx, "Amount"] = df_OCR.loc[next_amt, "Tekst"]

    # Date: wiersz z X0 w [30.19,30.20] -> najbliższy następny X0==81
    df_OCR["Date"] = None
    mask_date = (df_OCR["X0"] >= 30.19) & (df_OCR["X0"] <= 30.20)
    for idx in df_OCR[mask_date].index:
        next_81 = df_OCR.loc[idx + 1 :, "X0"][df_OCR["X0"] == 81].index.min()
        if pd.notna(next_81):
            df_OCR.loc[next_81, "Date"] = df_OCR.loc[idx, "Tekst"]

    # Porządki kolumn
    df_OCR.drop(["X0", "Y0", "X1", "Y1"], axis=1, inplace=True, errors="ignore")
    df_OCR.dropna(subset=["Text"], inplace=True)

    df_OCR["Partner name"] = df_OCR["Tekst"]
    df_OCR.drop(columns="Tekst", inplace=True, errors="ignore")
    df_OCR.rename(columns={"Strona": "Page"}, inplace=True)

    # Kolejność kolumn 1:1
    df_OCR = df_OCR[["Page", "Date", "Partner name", "Text", "Amount"]]

    # Amount -> Currency + Amount (ostatnie 3 znaki)
    df_OCR["Currency"] = df_OCR["Amount"].astype(str).str[-3:]
    df_OCR["Amount"] = df_OCR["Amount"].astype(str).str[:-3]

    # FV z Text (czyszczenie + prefiksy + regex + deduplikacja)
    df_OCR["FV"] = df_OCR["Text"].copy()
    df_OCR["FV"] = df_OCR["FV"].str.replace(" ", "", regex=True).str.upper()

    df_OCR["FV"] = df_OCR["FV"].apply(
        lambda x: x.replace("24270", "PL24270") if ("24270" in x and "PL" not in x) else x
    )
    df_OCR["FV"] = df_OCR["FV"].apply(
        lambda x: x.replace("24280", "DE24280") if ("24280" in x and "DE" not in x) else x
    )

    def extract_codes(v):
        if not isinstance(v, str):
            return None
        m = re.findall(r"(?:PL|DE)\d{10}", v)
        return " ".join(m) if m else None

    df_OCR["FV"] = df_OCR["FV"].apply(extract_codes)

    def dedup_space_separated(v):
        if pd.isna(v):
            return v
        parts = str(v).split()
        return " ".join(dict.fromkeys(parts))

    df_OCR["FV"] = df_OCR["FV"].apply(dedup_space_separated)

    # Finalna kolejność
    df_OCR = df_OCR[["Page", "Date", "Partner name", "Text", "Amount", "Currency", "FV"]]
    return df_OCR


def split_column_to_rows(df: pd.DataFrame, column_to_split: str) -> pd.DataFrame:
    """Rozbija wartości z kolumny (spacja-separowane) na osobne wiersze (jak w bazie)."""
    rows = []
    for _, row in df.iterrows():
        val = row.get(column_to_split)
        if pd.notna(val):
            for token in str(val).split():
                r = row.to_dict()
                r[column_to_split] = token
                rows.append(r)
        else:
            r = row.to_dict()
            r[column_to_split] = None
            rows.append(r)
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


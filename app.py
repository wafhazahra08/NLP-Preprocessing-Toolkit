import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Indonesian NLP Preprocessing Toolkit",
    layout="wide"
)

# ===============================
# Inisialisasi Stemmer (Sastrawi)
# ===============================
@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

# ===============================
# Stopwords Bahasa Indonesia (Sederhana)
# ===============================
stopwords_id = {
    "dan", "di", "ke", "dari", "yang", "untuk", "pada",
    "dengan", "adalah", "itu", "ini", "oleh", "sebagai"
}

# ===============================
# Kamus Normalisasi (Contoh)
# ===============================
normalization_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "aja": "saja",
    "udh": "sudah",
    "blm": "belum"
}

# ===============================
# Fungsi Preprocessing NLP
# ===============================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Case Folding
    text = text.lower()

    # 2. Tokenization & Cleaning
    tokens = re.findall(r'[a-z]+', text)

    # 3. Normalisasi
    normalized_tokens = [
        normalization_dict.get(token, token) for token in tokens
    ]

    # 4. Stopword Removal
    filtered_tokens = [
        token for token in normalized_tokens if token not in stopwords_id
    ]

    # 5. Stemming
    stemmed_tokens = [
        stemmer.stem(token) for token in filtered_tokens
    ]

    return " ".join(stemmed_tokens)

# ===============================
# Antarmuka Streamlit
# ===============================
st.title("🇮🇩 Indonesian NLP Preprocessing Toolkit")
st.write(
    """
    Aplikasi ini melakukan preprocessing teks Bahasa Indonesia yang mencakup:
    **case folding, tokenization, normalisasi, stopword removal dan stemming**.
    """
)

uploaded_file = st.file_uploader(
    "Unggah file CSV atau Excel",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    # Membaca file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview Data Asli")
    st.write(df.head())

    # Pilih Kolom Teks
    column_to_process = st.selectbox(
        "Pilih kolom teks yang akan diproses:",
        df.columns
    )

    if st.button("Mulai Preprocessing"):
        with st.spinner("Sedang melakukan preprocessing teks..."):
            df["teks_bersih"] = df[column_to_process].apply(preprocess_text)

        st.success("Preprocessing selesai!")

        st.subheader("Hasil Preprocessing")
        st.write(df[[column_to_process, "teks_bersih"]].head())

        # Download hasil
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh Hasil Preprocessing (CSV)",
            data=csv,
            file_name="hasil_preprocessing.csv",
            mime="text/csv"
        )

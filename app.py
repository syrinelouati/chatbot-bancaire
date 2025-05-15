# -*- coding: utf-8 -*-
import os
import json
import base64
import pandas as pd
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from groq import Groq

# --- Configuration Streamlit
st.set_page_config(page_title="Chatbot Bancaire + Extraction Virements", layout="centered")
st.title("💬 Chatbot & Extraction de Virements")

# Initialiser l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chargement modèle et données
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_data():
    df = pd.read_csv("cleanedTranslatedBankFAQs.csv", usecols=[
        "Question", "Answer", "Class", "Profile",
        "Profile_fr", "Profile_ar", "Class_fr", "Class_ar",
        "Question_fr", "Question_ar", "Answer_fr", "Answer_ar"
    ])
    return df

model = load_model()
df = load_data()

@st.cache_resource
def build_embeddings(df):
    embeddings = {
        "fr": model.encode(df["Profile_fr"].fillna('') + " - " + df["Question_fr"].fillna(''), show_progress_bar=True),
        "en": model.encode(df["Profile"].fillna('') + " - " + df["Question"].fillna(''), show_progress_bar=True),
        "ar": model.encode(df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna(''), show_progress_bar=True)
    }
    nn_models = {
        lang: NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[lang])
        for lang in embeddings
    }
    return embeddings, nn_models

embeddings, nn_models = build_embeddings(df)

# --- Initialiser client Groq
client = Groq(api_key="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")

# --- Fonctions OCR Générales
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_invoice_data_general(base64_image):
    system_prompt = """
    You are an OCR-like data extraction tool that extracts hotel invoice data from images.
    Extract data grouped by themes (e.g. invoice details, guest, rooms, taxes) and output JSON. 
    Maintain original language. Output blank/null fields if data is missing.
    """
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract invoice data."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

# --- Extraction spécifique des virements
def extract_invoice_data_virement(base64_image):
    system_prompt = """
    Extract invoice data and return JSON with these exact fields:
    - payer: {name: string, account: string (8 digits)}
    - payee: {name: string, account: string (20 digits)}
    - date: string (format DD/MM/YYYY)
    - amount: number
    - amount_words: string (French)
    - reason: string
    Return null for missing fields. Maintain this structure exactly.
    """
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all invoice data"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

# --- Conversion montant en lettres → chiffre
def convert_french_amount(words):
    french_numbers = {
        'zero': 0, 'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4,
        'cinq': 5, 'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9,
        'dix': 10, 'onze': 11, 'douze': 12, 'treize': 13,
        'quatorze': 14, 'quinze': 15, 'seize': 16,
        'dix-sept': 17, 'dix-huit': 18, 'dix-neuf': 19,
        'vingt': 20, 'trente': 30, 'quarante': 40,
        'cinquante': 50, 'soixante': 60, 'soixante-dix': 70,
        'quatre-vingt': 80, 'quatre-vingt-dix': 90,
        'cent': 100, 'cents': 100, 'mille': 1000
    }

    words = words.lower().replace('dinars', '').replace('dinar', '').strip()
    total = current = 0
    for word in words.split():
        if word in french_numbers:
            val = french_numbers[word]
            if val >= 100:
                current = 1 if current == 0 else current
                total += current * val
                current = 0
            else:
                current += val
    return total + current

# --- Chatbot multilingue
def generate_answer(user_question):
    try:
        lang = detect(user_question)
    except:
        lang = "fr"
    if lang not in ["fr", "en", "ar"]:
        lang = "fr"
    if lang == "fr":
        questions = df["Profile_fr"].fillna('') + " - " + df["Question_fr"].fillna('')
        answers = df["Answer_fr"]
        profils = df["Profile_fr"]
        intro_template = "🗣️ En tant que **{}**, tu peux :"
    elif lang == "ar":
        questions = df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna('')
        answers = df["Answer_ar"]
        profils = df["Profile_ar"]
        intro_template = "🗣️ بصفتك **{}**، يمكنك :"
    else:
        questions = df["Profile"].fillna('') + " - " + df["Question"].fillna('')
        answers = df["Answer"]
        profils = df["Profile"]
        intro_template = "🗣️ As a **{}**, you can:"

    input_embedding = model.encode([user_question])
    nn_model = nn_models[lang]
    _, index = nn_model.kneighbors(input_embedding)
    answer = answers.iloc[index[0][0]]
    profil = profils.iloc[index[0][0]]
    intro = intro_template.format(profil)
    return f"{intro}\n\n{answer}"

# --- Interface utilisateur Streamlit
mode = st.sidebar.selectbox("Choisir le mode :", ["Chatbot Bancaire", "Extraction Facture Générale", "Extraction Virement"])

if mode == "Chatbot Bancaire":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Posez votre question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

elif mode in ["Extraction Facture Générale", "Extraction Virement"]:
    st.write("📤 Upload une image (PNG, JPG, JPEG) à analyser.")
    uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Image uploadée", use_column_width=True)

        if st.button("Extraire les données"):
            with st.spinner("Extraction en cours..."):
                base64_img = encode_image(temp_path)
                try:
                    if mode == "Extraction Facture Générale":
                        json_data = extract_invoice_data_general(base64_img)
                        st.json(json.loads(json_data))
                    else:
                        json_data = extract_invoice_data_virement(base64_img)
                        parsed = json.loads(json_data)
                        st.json(parsed)

                        # Extra: Vérifier la cohérence montant ↔ montant en lettres
                        if parsed.get("amount_words"):
                            montant_calcule = convert_french_amount(parsed["amount_words"])
                            st.write(f"💡 Montant converti depuis le texte : **{montant_calcule}** Dinars")
                            if abs(montant_calcule - parsed.get("amount", 0)) > 1:
                                st.warning("❗ Le montant en chiffres ne correspond pas au montant en lettres.")
                except Exception as e:
                    st.error(f"Erreur lors de l'extraction : {e}")
            os.remove(temp_path)

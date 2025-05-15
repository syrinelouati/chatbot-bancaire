# -*- coding: utf-8 -*-
import os
import json
import base64
from datetime import datetime
from langdetect import detect
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from groq import Groq

# --- Initialisation
st.set_page_config(page_title="Chatbot & Extraction Virements", layout="centered")
st.title("🏦 Chatbot Bancaire")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chargement données & modèles
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
        "fr": model.encode(df["Profile_fr"].fillna('') + " - " + df["Question_fr"].fillna('')),
        "en": model.encode(df["Profile"].fillna('') + " - " + df["Question"].fillna('')),
        "ar": model.encode(df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna(''))
    }
    nn_models = {
        lang: NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[lang])
        for lang in embeddings
    }
    return embeddings, nn_models

embeddings, nn_models = build_embeddings(df)

# Client Groq
client = Groq(api_key="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")

# --- Encodage image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Extraction de données de virement
def extract_invoice_data_virement(base64_image):
    prompt = """
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
            {"role": "system", "content": prompt},
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
    return json.loads(response.choices[0].message.content)

# --- Conversion texte montant → nombre
def convert_french_amount(words):
    words = words.lower().replace('dinars', '').replace('dinar', '').strip()
    french_numbers = {
        'zero': 0, 'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4,
        'cinq': 5, 'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9,
        'dix': 10, 'onze': 11, 'douze': 12, 'treize': 13, 'quatorze': 14,
        'quinze': 15, 'seize': 16, 'dix-sept': 17, 'dix-huit': 18,
        'dix-neuf': 19, 'vingt': 20, 'trente': 30, 'quarante': 40,
        'cinquante': 50, 'soixante': 60, 'soixante-dix': 70,
        'quatre-vingt': 80, 'quatre-vingt-dix': 90,
        'cent': 100, 'cents': 100, 'mille': 1000
    }

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

# --- Validation des champs
def validate_data(data):
    errors = []

    payer_account = data.get("payer", {}).get("account", "")
    if not (payer_account.isdigit() and len(payer_account) == 8):
        errors.append("Compte payer invalide (8 chiffres requis)")

    payee_account = data.get("payee", {}).get("account", "")
    if not (payee_account.isdigit() and len(payee_account) == 20):
        errors.append("Compte payee invalide (20 chiffres requis)")

    try:
        datetime.strptime(data.get("date", ""), "%d/%m/%Y")
    except:
        errors.append("Date invalide (format attendu : JJ/MM/AAAA)")

    if "amount_words" in data:
        montant_lettres = convert_french_amount(data["amount_words"])
        montant_chiffre = data.get("amount", 0)
        if abs(montant_chiffre - montant_lettres) > 1:
            errors.append("Montant en lettres ne correspond pas au montant en chiffres")

    return errors

# --- Chatbot
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
        intro = "🗣️ En tant que **{}**, tu peux :"
    elif lang == "ar":
        questions = df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna('')
        answers = df["Answer_ar"]
        profils = df["Profile_ar"]
        intro = "🗣️ بصفتك **{}**، يمكنك :"
    else:
        questions = df["Profile"].fillna('') + " - " + df["Question"].fillna('')
        answers = df["Answer"]
        profils = df["Profile"]
        intro = "🗣️ As a **{}**, you can:"

    input_embedding = model.encode([user_question])
    _, index = nn_models[lang].kneighbors(input_embedding)
    return f"{intro.format(profils.iloc[index[0][0]])}\n\n{answers.iloc[index[0][0]]}"

# --- Interface
mode = st.sidebar.radio("Choisir le mode :", ["Chatbot", "Extraction Virement"])

if mode == "Chatbot":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Pose ta question bancaire..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = generate_answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

else:
    uploaded_file = st.file_uploader("🖼️ Charge une image de virement", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        tmp_path = f"tmp_{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Image analysée", use_column_width=True)

        if st.button("📄 Extraire et valider"):
            with st.spinner("Analyse en cours..."):
                try:
                    base64_img = encode_image(tmp_path)
                    result = extract_invoice_data_virement(base64_img)
                    st.subheader("📋 Données extraites")
                    st.json(result)

                    errors = validate_data(result)
                    if not errors:
                        st.success("✅ Données valides")
                    else:
                        st.error("❌ Données invalides :")
                        for err in errors:
                            st.markdown(f"- {err}")

                except Exception as e:
                    st.error(f"Erreur : {e}")
            os.remove(tmp_path)

# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from langdetect import detect
import streamlit as st
import os

# Fix Streamlit file watcher
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Configuration de l'application Streamlit
st.set_page_config(page_title="Chatbot Bancaire", layout="centered")
st.title("💬 Chatbot Bancaire Multilingue")

# Initialiser l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chargement du modèle
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# Chargement du dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleanedTranslatedBankFAQs.csv", encoding='utf-8', dtype=str).fillna('')
    except UnicodeDecodeError:
        df = pd.read_csv("cleanedTranslatedBankFAQs.csv", encoding='utf-8-sig', dtype=str).fillna('')
    return df

df = load_data()

# Création des embeddings pour chaque langue
@st.cache_resource
def build_embeddings(df):
    embeddings = {
        "fr": model.encode(df["Profile_fr"] + " - " + df["Question_fr"], show_progress_bar=True),
        "en": model.encode(df["Profile"] + " - " + df["Question"], show_progress_bar=True),
        "ar": model.encode(df["Profile_ar"] + " - " + df["Question_ar"], show_progress_bar=True)
    }
    nn_models = {
        lang: NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[lang])
        for lang in embeddings
    }
    return embeddings, nn_models

embeddings, nn_models = build_embeddings(df)

# Fonction principale de génération de réponse
def generate_answer(user_question):
    try:
        lang = detect(user_question)
    except:
        lang = "fr"

    if lang not in ["fr", "en", "ar"]:
        lang = "fr"

    # Sélection des colonnes selon la langue
    if lang == "fr":
        questions = df["Profile_fr"] + " - " + df["Question_fr"]
        answers = df["Answer_fr"]
        profils = df["Profile_fr"]
        intro_template = "🗣️ En tant que **{}**, tu peux :"
    elif lang == "ar":
        questions = df["Profile_ar"] + " - " + df["Question_ar"]
        answers = df["Answer_ar"]
        profils = df["Profile_ar"]
        intro_template = "🗣️ بصفتك **{}**، يمكنك :"
    else:
        questions = df["Profile"] + " - " + df["Question"]
        answers = df["Answer"]
        profils = df["Profile"]
        intro_template = "🗣️ As a **{}**, you can:"

    # Encodage de la question de l'utilisateur
    input_embedding = model.encode([user_question])
    nn_model = nn_models[lang]
    distance, index = nn_model.kneighbors(input_embedding)

    answer = answers.iloc[index[0][0]]
    profil = profils.iloc[index[0][0]]
    intro = intro_template.format(profil)

    # Assurer un affichage correct des symboles monétaires
    return f"{intro}\n\n{answer}"

# Affichage de l'historique du chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Champ de saisie
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

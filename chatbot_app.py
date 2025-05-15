# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from langdetect import detect
import streamlit as st
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Titre de l'app
st.title("🤖 Chatbot Bancaire Multilingue")

# Chargement du modèle de phrases
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("cleanedTranslatedBankFAQs.csv", usecols=[
        "Question", "Answer", "Class", "Profile",
        "Profile_fr", "Profile_ar", "Class_fr", "Class_ar",
        "Question_fr", "Question_ar", "Answer_fr", "Answer_ar"
    ])
    return df

df = load_data()

# Préparation des embeddings pour les 3 langues
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

# Interface utilisateur
user_question = st.text_area("❓ Posez votre question bancaire")

if st.button("Obtenir la réponse"):
    if user_question:
        try:
            lang = detect(user_question)
        except:
            lang = "fr"  # fallback si détection échoue

        if lang not in ["fr", "en", "ar"]:
            lang = "fr"  # par défaut

        # Sélection des bonnes colonnes et embeddings
        if lang == "fr":
            questions = df["Profile_fr"].fillna('') + " - " + df["Question_fr"].fillna('')
            answers = df["Answer_fr"]
            profils = df["Profile_fr"]
            classes = df["Class_fr"]
        elif lang == "ar":
            questions = df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna('')
            answers = df["Answer_ar"]
            profils = df["Profile_ar"]
            classes = df["Class_ar"]
        else:
            questions = df["Profile"].fillna('') + " - " + df["Question"].fillna('')
            answers = df["Answer"]
            profils = df["Profile"]
            classes = df["Class"]

        # Recherche de la réponse la plus proche
        input_embedding = model.encode([user_question])
        nn_model = nn_models[lang]
        distance, index = nn_model.kneighbors(input_embedding)

        answer = answers.iloc[index[0][0]]
        profil = profils.iloc[index[0][0]]
        class_name = classes.iloc[index[0][0]]

        # Introduction en fonction de la langue
        if lang == 'fr':
            intro = f"🗣️ En tant que **{profil}**, tu peux :"
            class_label = f"📂 Classe : {class_name}"
        elif lang == 'ar':
            intro = f"🗣️ بصفتك **{profil}**، يمكنك :"
            class_label = f"📂 الفئة : {class_name}"
        else:
            intro = f"🗣️ As a **{profil}**, you can:"
            class_label = f"📂 Class: {class_name}"

        st.success(f"{intro}\n\n{answer}")
        st.info(class_label)
    else:
        st.warning("⚠️ Veuillez poser une question.")

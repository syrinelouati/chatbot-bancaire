# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from langdetect import detect
import streamlit as st
import os

# Pour éviter les erreurs de surveillance des fichiers sur Streamlit Cloud
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="Chatbot Bancaire Multilingue", page_icon="🤖")
st.title("🤖 Chatbot Bancaire Multilingue")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("cleanedTranslatedBankFAQs.csv", usecols=[
        "Question", "Answer", "Class", "Profile",
        "Profile_fr", "Profile_ar", "Class_fr", "Class_ar",
        "Question_fr", "Question_ar", "Answer_fr", "Answer_ar"
    ])
    df["full_question"] = df["Profile"].fillna('') + " - " + df["Question"].fillna('')
    return df

df = load_data()

# Encodage et modèle
@st.cache_resource
def build_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["full_question"].tolist(), show_progress_bar=True)
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(embeddings)
    return model, nn_model, embeddings

model, nn_model, question_embeddings = build_model_and_embeddings(df)

# Interface utilisateur
user_input = st.text_area("❓ Posez votre question bancaire avec votre profil (ex : client - comment ouvrir un compte ?)")

if st.button("Obtenir la réponse"):
    if user_input:
        try:
            # Détection de la langue
            lang = detect(user_input)

            # Recherche de la meilleure correspondance
            input_embedding = model.encode([user_input])
            distance, index = nn_model.kneighbors(input_embedding)
            row = df.iloc[index[0][0]]

            # Récupération du profil et de la réponse adaptés à la langue
            if lang == 'fr':
                profil = row["Profile_fr"]
                answer = row["Answer_fr"]
                classe = row["Class_fr"]
            elif lang == 'ar':
                profil = row["Profile_ar"]
                answer = row["Answer_ar"]
                classe = row["Class_ar"]
            else:
                profil = row["Profile"]
                answer = row["Answer"]
                classe = row["Class"]

            # Affichage
            st.success(f"🗣️ En tant que **{profil}**, tu peux :\n\n{answer}")
            st.info(f"📂 Classe : {classe}")

        except Exception as e:
            st.error("Une erreur est survenue lors du traitement de votre question.")
    else:
        st.warning("⚠️ Veuillez poser une question incluant votre profil (ex : client - je veux ouvrir un compte).")

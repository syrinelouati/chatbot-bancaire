# -*- coding: utf-8 -*-
"""chatbot_app"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from langdetect import detect
import streamlit as st
import os

# Pour éviter certains bugs avec Streamlit
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Titre
st.title("🤖 Chatbot Bancaire Multilingue")

# Chargement du modèle
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("cleanedTranslatedBankFAQs.csv")
    return df

df = load_data()

# Saisie de la question
user_question = st.text_area("❓ Posez votre question bancaire")

# Traitement si une question est saisie
if user_question:
    try:
        langue = detect(user_question)
    except:
        langue = "en"

    # Définir les colonnes en fonction de la langue
    if langue.startswith("fr"):
        col_question = "Question_fr"
        col_answer = "Answer_fr"
        col_profile = "Profile_fr"
        col_class = "Class_fr"
    elif langue.startswith("ar"):
        col_question = "Question_ar"
        col_answer = "Answer_ar"
        col_profile = "Profile_ar"
        col_class = "Class_ar"
    else:
        col_question = "Question"
        col_answer = "Answer"
        col_profile = "Profile"
        col_class = "Class"

    # Liste des profils disponibles (uniques) dans la langue détectée
    profils_disponibles = df[col_profile].dropna().unique().tolist()
    profils_disponibles.sort()

    # Sélection du profil via menu déroulant
    user_profile = st.selectbox("👤 Sélectionnez votre profil", options=profils_disponibles)

    if st.button("Obtenir la réponse"):
        # Création des textes combinés pour l'encodage
        df["full_question"] = df[col_profile].fillna('') + " - " + df[col_question].fillna('')

        # Filtrage par profil sélectionné
        df_filtered = df[df[col_profile] == user_profile]

        if df_filtered.empty:
            st.warning("⚠️ Aucun résultat trouvé pour ce profil dans cette langue.")
        else:
            # Encodage des questions
            question_embeddings = model.encode(df_filtered["full_question"].tolist(), convert_to_tensor=True)
            input_embedding = model.encode([user_profile + " - " + user_question], convert_to_tensor=True)

            # Similarité cosinus
            scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0]
            best_index = scores.argmax().item()

            # Résultat
            response = df_filtered.iloc[best_index][col_answer]
            classe = df_filtered.iloc[best_index][col_class]

            st.success(f"💬 **Réponse :** {response}")
            st.info(f"📂 **Classe :** {classe}")
else:
    st.info("💡 Veuillez d'abord poser une question pour détecter la langue et activer la sélection du profil.")

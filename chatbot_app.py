# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from langdetect import detect
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Titre
st.title("🤖 Chatbot Bancaire Multilingue")

# Chargement du modèle
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("cleanedTranslatedBankFAQs.csv")
    df["full_question"] = df["Profile"].fillna('') + " - " + df["Question"].fillna('')
    return df

df = load_data()

# Encoder les questions
@st.cache_resource
def build_model_and_embeddings(df):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["full_question"].tolist(), show_progress_bar=True)
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(embeddings)
    return model, nn_model, embeddings

model, nn_model, question_embeddings = build_model_and_embeddings(df)

# Interface utilisateur
st.markdown("Posez votre question bancaire. Le chatbot s'adapte automatiquement à votre langue.")
user_question = st.text_area("❓ Posez votre question")

if st.button("Obtenir la réponse"):
    if user_question.strip():
        # Détecter la langue de la question
        lang = detect(user_question)

        # Rechercher la meilleure réponse dans les données
        best_score = -1
        best_index = -1

        # Essayer avec tous les profils pour trouver la meilleure correspondance
        for i, row in df.iterrows():
            input_text = f"{row['Profile']} - {user_question}"
            input_embedding = model.encode([input_text])
            distance, _ = nn_model.kneighbors(input_embedding)
            similarity = 1 - distance[0][0]  # plus c'est proche de 1, mieux c'est

            if similarity > best_score:
                best_score = similarity
                best_index = i

        if best_index != -1:
            matched_row = df.iloc[best_index]
            profile = matched_row["Profile"]
            class_name = matched_row["Class"]

            if lang == "fr":
                response = matched_row["Answer_fr"]
                profile_lang = matched_row["Profile_fr"]
                final_response = f"En tant que {profile_lang}, tu peux {response}"
            elif lang == "ar":
                response = matched_row["Answer_ar"]
                profile_lang = matched_row["Profile_ar"]
                final_response = f"بصفتك {profile_lang}، يمكنك {response}"
            else:
                response = matched_row["Answer"]
                final_response = f"As a {profile}, you can {response}"

            st.success(f"**Réponse :** {final_response}")
            st.info(f"📂 **Classe détectée :** {class_name}")
        else:
            st.warning("❌ Aucune réponse pertinente trouvée.")
    else:
        st.warning("⚠️ Veuillez poser une question.")

# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from langdetect import detect
import streamlit as st
import os
 
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
 
st.set_page_config(page_title="Chatbot Bancaire", layout="centered")
st.title("💬 Chatbot Bancaire Multilingue")
 
# Initialiser l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Chargement du modèle
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
 
# Embeddings
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
 
# Fonction de réponse
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
        classes = df["Class_fr"]
        intro_template = "🗣️ En tant que **{}**, tu peux :"
    elif lang == "ar":
        questions = df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna('')
        answers = df["Answer_ar"]
        profils = df["Profile_ar"]
        classes = df["Class_ar"]
        intro_template = "🗣️ بصفتك **{}**، يمكنك :"
    else:
        questions = df["Profile"].fillna('') + " - " + df["Question"].fillna('')
        answers = df["Answer"]
        profils = df["Profile"]
        classes = df["Class"]
        intro_template = "🗣️ As a **{}**, you can:"
 
    input_embedding = model.encode([user_question])
    nn_model = nn_models[lang]
    distance, index = nn_model.kneighbors(input_embedding)
 
    answer = answers.iloc[index[0][0]]
    profil = profils.iloc[index[0][0]]
    intro = intro_template.format(profil)
 
    return f"{intro}\n\n{answer}"
 
# Afficher l'historique des messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
 
# Saisie de l'utilisateur
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter la question à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # Générer la réponse
    response = generate_answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

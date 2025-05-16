# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from langdetect import detect
import streamlit as st
import os
import re

# Packages pour conversion mots -> nombres (penser à installer)
try:
    from text_to_num import text2num  # pour fr/ar
except ImportError:
    text2num = None

try:
    from word2number import w2n  # pour en
except ImportError:
    w2n = None

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="Chatbot Bancaire", layout="centered")
st.title("💬 Chatbot Bancaire Multilingue avec validation de montant")

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

# Fonction de conversion et validation du montant écrit en toutes lettres
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)  # retirer ponctuation
    text = text.strip()
    return text

def convert_words_to_num(text, lang):
    if lang in ["fr", "ar"]:
        if text2num is None:
            raise ImportError("Le package 'text_to_num' est requis pour français et arabe. Installer via 'pip install text_to_num'.")
        return text2num(text, lang)
    elif lang == "en":
        if w2n is None:
            raise ImportError("Le package 'word2number' est requis pour anglais. Installer via 'pip install word2number'.")
        return w2n.word_to_num(text)
    else:
        raise ValueError(f"Langue non prise en charge pour la conversion : {lang}")

def validate_amount_general(amount_numeric, amount_words):
    try:
        lang = detect(amount_words)
    except:
        lang = "fr"  # défaut

    amount_words_norm = normalize_text(amount_words)

    try:
        amount_converted = convert_words_to_num(amount_words_norm, lang)
    except Exception as e:
        return False, f"Erreur conversion montant mots -> nombre: {e}"

    if abs(amount_converted - amount_numeric) < 0.01:  # tolérance
        return True, amount_converted
    else:
        return False, amount_converted

# Fonction de réponse du chatbot
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

    # Exemple d'utilisation de validation montant
    # (Tu peux adapter cette partie pour extraire montant_numeric et montant_en_mots depuis la question ou données)
    # Pour démonstration, supposons un montant attendu et sa forme en mots extraits :
    montant_attendu = 6889
    montant_mots_exemple = "Six mille huit cent quatre-vingt-neuf dinars"

    valid, converted = validate_amount_general(montant_attendu, montant_mots_exemple)
    if valid:
        validation_msg = "✅ Montant valide (montant numérique correspond bien au montant écrit en toutes lettres)."
    else:
        validation_msg = f"❌ Montant invalide : attendu {montant_attendu}, mais montant converti {converted}"

    return f"{intro}\n\n{answer}\n\n\nValidation montant exemple :\n{validation_msg}"

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

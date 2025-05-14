import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from langdetect import detect
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.title("🤖 Chatbot Bancaire Multilingue")

# Charger uniquement les colonnes utiles
@st.cache_data
def load_data():
    df = pd.read_csv("cleanedTranslatedBankFAQs.csv", usecols=[
        "Question", "Answer", "Class", "Profile",
        "Profile_fr", "Profile_ar",
        "Answer_fr", "Answer_ar",
        "full_question"
    ])
    df["full_question"] = df["Profile"].fillna('') + " - " + df["Question"].fillna('')
    return df

df = load_data()

# Charger les embeddings pré-calculés
@st.cache_resource
def build_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["full_question"].tolist(), show_progress_bar=False)
    nn_model = NearestNeighbors(n_neighbors=3, metric="cosine")  # 3 voisins max
    nn_model.fit(embeddings)
    return model, nn_model, embeddings

model, nn_model, embeddings = build_model_and_embeddings(df)

# Interface utilisateur
user_question = st.text_area("❓ Posez votre question")

if st.button("Obtenir la réponse"):
    if user_question.strip():
        try:
            lang = detect(user_question)
        except:
            lang = "fr"  # défaut si détection échoue

        # Encoder la question une seule fois
        input_embedding = model.encode([user_question])
        distances, indices = nn_model.kneighbors(input_embedding)

        # Prendre la réponse la plus proche
        best_index = indices[0][0]
        matched_row = df.iloc[best_index]

        # Détecter la langue et formuler la réponse adaptée
        if lang == "fr":
            response = matched_row["Answer_fr"]
            profile = matched_row["Profile_fr"]
            final_response = f"En tant que {profile}, tu peux {response}"
        elif lang == "ar":
            response = matched_row["Answer_ar"]
            profile = matched_row["Profile_ar"]
            final_response = f"بصفتك {profile}، يمكنك {response}"
        else:
            response = matched_row["Answer"]
            profile = matched_row["Profile"]
            final_response = f"As a {profile}, you can {response}"

        st.success(f"**Réponse :** {final_response}")
        st.info(f"📂 **Classe détectée :** {matched_row['Class']}")
    else:
        st.warning("⚠️ Veuillez poser une question.")

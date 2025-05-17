# -*- coding: utf-8 -*-
import os
import json
import base64
import pandas as pd
from datetime import datetime
from PIL import Image
import io

import streamlit as st
from groq import Groq
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Configuration de la page Streamlit
st.set_page_config(page_title="Chatbot Bancaire + Extraction Virements", layout="wide")
st.title("💬 Chatbot Bancaire")

# === INITIALISATION CHATBOT ===
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_data():
    return pd.read_csv("cleanedTranslatedBankFAQs.csv", usecols=[
        "Question", "Answer", "Class", "Profile",
        "Profile_fr", "Profile_ar", "Class_fr", "Class_ar",
        "Question_fr", "Question_ar", "Answer_fr", "Answer_ar"
    ])

model = load_model()
df = load_data()

@st.cache_resource
def build_embeddings(df):
    embeddings = {
        "fr": model.encode(df["Profile_fr"].fillna('') + " - " + df["Question_fr"].fillna('')),
        "en": model.encode(df["Profile"].fillna('') + " - " + df["Question"].fillna('')),
        "ar": model.encode(df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna(''))
    }
    
    #Entraînement d'un modèle de plus proche voisin (KNN) par langue pour retrouver la question la plus similaire
    nn_models = {
        lang: NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[lang])
        for lang in embeddings
    }
    return embeddings, nn_models

embeddings, nn_models = build_embeddings(df)

# === INITIALISATION EXTRACTION virement ===

#Initialise le client avec la clé API Groq pour utiliser le modèle LLaMA.
client = Groq(api_key="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")

def encode_image_file(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

def extract_payement_data(base64_image):
    system_prompt = """
    Extract payement data and return JSON with these exact fields:
    - payer: {name: string, account: string (8 digits)}
    - payee: {name: string, account: string (20 digits)}
    - date: string (format DD/MM/YYYY)
    - amount_words: string (French)
    - reason: string

    Return null for missing fields. Maintain this structure exactly.
    """

#Appelle LLaMA 4 via Groq pour extraire les données d’un virement bancaire à partir d’une image base64
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
    return json.loads(response.choices[0].message.content)

#Convertit un montant écrit en lettres françaises en nombre
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


#- Transforme la chaîne en **minuscules**.
#- Supprime les mots "dinar" ou "dinars" s'ils sont présents.
#- Supprime les espaces en début/fin.
    
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

#Vérifie le format de la date
def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%d/%m/%Y")
        return True
    except:
        return False

def validate_name(name, name_type):
    """Validate payer/payee name"""
    if not name or not str(name).strip():
        return f"❌ Missing {name_type} name"
    if len(str(name).strip()) < 2:
        return f"❌ {name_type} name too short"
    return f"✅ Valid {name_type} name"
 
def validate_and_display(data):
    """Perform all validations and display results"""
    print("\n--- EXTRACTED DATA ---")
    for section, content in data.items():
        print(f"\n{section.title()}:")
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"  {k.title()}: {v if v is not None else 'null'}")
        else:
            print(f"  {content if content is not None else 'null'}")
 
    print("\n--- VALIDATION RESULTS ---")
 
    # 1. Name validation
    payer_name_valid = validate_name(data.get('payer', {}).get('name'), 'payer')
    payee_name_valid = validate_name(data.get('payee', {}).get('name'), 'payee')
 
    # 2. Account number validation
    def validate_account(account, expected_len, acc_type):
        if not account:
            return f"❌ Missing {acc_type} account"
        if len(str(account)) != expected_len:
            return f"❌ Invalid {acc_type} account length (expected {expected_len}, got {len(str(account))})"
        return f"✅ Valid {acc_type} account"
 
    payer_account_valid = validate_account(data.get('payer', {}).get('account'), 8, 'payer')
    payee_account_valid = validate_account(data.get('payee', {}).get('account'), 20, 'payee')
 
    # 3. Date validation
    date_valid = (
        "✅ Valid date" if validate_date(data.get('date', ''))
        else "❌ Invalid or missing date (required format: DD/MM/YYYY)"
    )
 
    # 4. Amount validation
    # amount_valid = "❌ Missing amount information"
    # if 'amount' in data and 'amount_words' in data:
    #     try:
    #         converted = convert_french_amount(data['amount_words'])
    #         if float(data['amount']) == converted:
    #             amount_valid = "✅ Amount matches"
    #         else:
    #             amount_valid = f"❌ Amount mismatch "
    #     except Exception as e:
    #         amount_valid = f"❌ Amount validation error: {str(e)}"
 
    # Print all validation results
    print(f"• {payer_name_valid}")
    print(f"• {payee_name_valid}")
    print(f"• {payer_account_valid}")
    print(f"• {payee_account_valid}")
    print(f"• {date_valid}")
    # print(f"• {amount_valid}")
 
def process_invoice(image_path, output_dir):
    """Process single invoice image"""
    try:
        print(f"\nProcessing: {os.path.basename(image_path)}")
 
        # Extract data
        base64_img = encode_image(image_path)
        invoice_json = extract_invoice_data(base64_img)
        data = json.loads(invoice_json)
 
        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            os.path.basename(image_path).replace(".jpg", ".json").replace(".png", ".json")
        )
 
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
 
        # Validate and display
        validate_and_display(data)
        return output_path
 
    except Exception as e:
        print(f"Error processing invoice: {str(e)}")
        return None
 

# === INTERFACE STREAMLIT ===
tab1, tab2 = st.tabs(["📩 Chatbot Bancaire", "📤 Extraction Virements"])

with tab1:
    st.subheader("💬 Assistant Bancaire Intelligent")

    # Greeting selon l'heure
    now = datetime.now().hour
    if now < 12:
        greeting = "☀️ Bonjour !"
    elif now < 18:
        greeting = "🌤️ Bon après-midi !"
    else:
        greeting = "🌙 Bonsoir !"

    st.markdown(f"### {greeting} Comment puis-je vous aider aujourd’hui ?")

    # Zone de question
    user_input = st.text_input("💡 Posez une question bancaire ci-dessous :")

    if user_input:
        lang = detect(user_input)
        query = model.encode(user_input)
        distances, indices = nn_models[lang].kneighbors([query])
        idx = indices[0][0]
        st.write("### 📌 Réponse suggérée :")
        st.success(df.iloc[idx][f"Answer_{lang}" if lang != "en" else "Answer"])

with tab2:
    st.subheader("Uploader un virement à analyser")
    uploaded_file = st.file_uploader("📎 Déposez une image (.png/.jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        base64_img = encode_image_file(uploaded_file)
        st.image(uploaded_file, caption="Virement uploadée", use_column_width=True)
        with st.spinner("🧠 Extraction en cours..."):
            extracted_data = extract_payement_data(base64_img)
            st.json(extracted_data)
            st.markdown("### ✅ Résultats de validation")
            for check in validate_and_display(extracted_data):
                st.write(f"- {check}")
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=60)

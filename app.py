from groq import Groq
import os
import json
import base64
from datetime import datetime
from PIL import Image
import io

# Initialize Groq client
client = Groq(api_key="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")

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_transfer_data(base64_image):
    """Extract bank transfer data with specific field requirements"""
    system_prompt = """
    Tu es un expert en lecture automatique de documents bancaires. Extrait les données d’un virement bancaire à partir d’une image et retourne un JSON contenant exactement les champs suivants :
    - payer: {name: string, account: string (8 chiffres)}
    - payee: {name: string, account: string (20 chiffres)}
    - date: string (format DD/MM/YYYY)
    - amount: number
    - amount_words: string (en toutes lettres en français)
    - reason: string
    Si une information est manquante, retourne "null" pour cette information. Garde exactement cette structure.
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extrait les données du virement bancaire ci-joint"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def convert_french_amount(words):
    """Convert French number words to numeric value"""
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

def validate_date(date_str):
    """Validate date format (DD/MM/YYYY)"""
    try:
        datetime.strptime(date_str, "%d/%m/%Y")
        return True
    except (ValueError, TypeError):
        return False

def validate_and_display(data):
    """Perform all validations and display results"""
    print("\n--- DONNÉES EXTRACTÉES ---")
    for section, content in data.items():
        print(f"\n{section.title()}:")
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"  {k.title()}: {v if v is not None else 'null'}")
        else:
            print(f"  {content if content is not None else 'null'}")

    print("\n--- RÉSULTATS DE VALIDATION ---")

    def validate_account(account, expected_len, acc_type):
        if not account:
            return f"❌ {acc_type} manquant"
        if len(str(account)) != expected_len:
            return f"❌ Longueur invalide pour {acc_type} (attendue: {expected_len}, obtenue: {len(str(account))})"
        return f"✅ {acc_type} valide"

    payer_valid = validate_account(data.get('payer', {}).get('account'), 8, 'compte payeur')
    payee_valid = validate_account(data.get('payee', {}).get('account'), 20, 'compte bénéficiaire')

    date_valid = (
        "✅ Date valide" if validate_date(data.get('date', ''))
        else "❌ Date invalide ou manquante (format attendu: JJ/MM/AAAA)"
    )

    amount_valid = "❌ Informations de montant manquantes"
    if 'amount' in data and 'amount_words' in data:
        try:
            converted = convert_french_amount(data['amount_words'])
            if float(data['amount']) == converted:
                amount_valid = "✅ Montant cohérent"
            else:
                amount_valid = f"❌ Montant incohérent (chiffre: {data['amount']}, en lettres: {converted})"
        except Exception as e:
            amount_valid = f"❌ Erreur de validation du montant: {str(e)}"

    print(f"• {payer_valid}")
    print(f"• {payee_valid}")
    print(f"• {date_valid}")
    print(f"• {amount_valid}")

def process_transfer(image_path, output_dir):
    """Process a single bank transfer image"""
    try:
        print(f"\nTraitement de : {os.path.basename(image_path)}")

        # Encode and extract
        base64_img = encode_image(image_path)
        transfer_json = extract_transfer_data(base64_img)
        data = json.loads(transfer_json)

        # Save result
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            os.path.basename(image_path).replace(".jpg", ".json").replace(".png", ".json")
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # Validate and print
        validate_and_display(data)
        return output_path

    except Exception as e:
        print(f"Erreur lors du traitement du virement : {str(e)}")
        return None

# Configuration
input_dir = "/content"
output_dir = "/content/extracted_data"

# Process all transfer images
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_transfer(os.path.join(input_dir, filename), output_dir)

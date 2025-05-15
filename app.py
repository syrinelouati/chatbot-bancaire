from groq import Groq
import os
import json
import base64
from datetime import datetime

# Initialize Groq client
client = Groq(api_key="TON_API_KEY_ICI")  # Ne jamais exposer une vraie clé API en public

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_invoice_data(base64_image):
    system_prompt = """
    Extract invoice data and return JSON with these exact fields:
    - payer: {name: string, account: string (8 digits)}
    - payee: {name: string, account: string (20 digits)}
    - date: string (format DD/MM/YYYY)
    - amount: number
    - amount_words: string (French)
    - reason: string
    Return null for missing fields. Maintain this structure exactly.
    """

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
    return response.choices[0].message.content

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

def validate_name(name, name_type):
    if not name or not str(name).strip():
        return f"❌ Missing {name_type} name"
    if len(str(name).strip()) < 2:
        return f"❌ {name_type} name too short"
    return f"✅ Valid {name_type} name"

def validate_account(account, expected_len, acc_type):
    if not account:
        return f"❌ Missing {acc_type} account"
    if len(str(account)) != expected_len:
        return f"❌ Invalid {acc_type} account length (expected {expected_len}, got {len(str(account))})"
    return f"✅ Valid {acc_type} account"

def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%d/%m/%Y")
        return True
    except (ValueError, TypeError):
        return False

def validate_data(data):
    print("\n--- VALIDATION ---")

    print(f"• {validate_name(data.get('payer', {}).get('name'), 'payer')}")
    print(f"• {validate_name(data.get('payee', {}).get('name'), 'payee')}")
    print(f"• {validate_account(data.get('payer', {}).get('account'), 8, 'payer')}")
    print(f"• {validate_account(data.get('payee', {}).get('account'), 20, 'payee')}")

    date = data.get('date', '')
    print("• ✅ Valid date" if validate_date(date) else "• ❌ Invalid or missing date")

    if 'amount' in data and 'amount_words' in data:
        try:
            converted = convert_french_amount(data['amount_words'])
            if float(data['amount']) == converted:
                print("• ✅ Amount matches")
            else:
                print(f"• ❌ Amount mismatch (value: {data['amount']}, words: {converted})")
        except Exception as e:
            print(f"• ❌ Amount validation error: {str(e)}")
    else:
        print("• ❌ Missing amount or amount_words")

def process_invoice(image_path, output_dir):
    try:
        base64_img = encode_image(image_path)
        invoice_json = extract_invoice_data(base64_img)
        data = json.loads(invoice_json)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            os.path.basename(image_path).replace(".jpg", ".json").replace(".png", ".json")
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        validate_data(data)
        return output_path
    except Exception as e:
        print(f"Error processing invoice: {str(e)}")
        return None

# Configuration
input_dir = "/content"
output_dir = "/content/extracted_data"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_invoice(os.path.join(input_dir, filename), output_dir)

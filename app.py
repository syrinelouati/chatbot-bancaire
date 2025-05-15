from groq import Groq
import os
import json
import base64
from datetime import datetime

# Initialize Groq client
client = Groq(api_key="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")

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_invoice_data(base64_image):
    """Extract invoice data with specific field requirements"""
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
    print("\n--- EXTRACTED DATA ---")
    for section, content in data.items():
        print(f"\n{section.title()}:")
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"  {k.title()}: {v if v is not None else 'null'}")
        else:
            print(f"  {content if content is not None else 'null'}")

    print("\n--- VALIDATION RESULTS ---")

    def validate_account(account, expected_len, acc_type):
        if not account:
            return f"❌ Missing {acc_type} account"
        if len(str(account)) != expected_len:
            return f"❌ Invalid {acc_type} account length (expected {expected_len}, got {len(str(account))})"
        return f"✅ Valid {acc_type} account"

    payer_valid = validate_account(data.get('payer', {}).get('account'), 8, 'payer')
    payee_valid = validate_account(data.get('payee', {}).get('account'), 20, 'payee')

    date_valid = (
        "✅ Valid date" if validate_date(data.get('date', ''))
        else "❌ Invalid or missing date (required format: DD/MM/YYYY)"
    )

    amount_valid = "❌ Missing amount information"
    if 'amount' in data and 'amount_words' in data:
        try:
            converted = convert_french_amount(data['amount_words'])
            if float(data['amount']) == converted:
                amount_valid = "✅ Amount matches"
            else:
                amount_valid = f"❌ Amount mismatch (numbers: {data['amount']}, words convert to: {converted})"
        except Exception as e:
            amount_valid = f"❌ Amount validation error: {str(e)}"

    print(f"• {payer_valid}")
    print(f"• {payee_valid}")
    print(f"• {date_valid}")
    print(f"• {amount_valid}")

def process_invoice(image_path, output_dir):
    """Process single invoice image"""
    try:
        print(f"\nProcessing: {os.path.basename(image_path)}")

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

        validate_and_display(data)
        return output_path

    except Exception as e:
        print(f"Error processing invoice: {str(e)}")
        return None

if __name__ == "__main__":
    input_dir = "/content"
    output_dir = "/content/extracted_data"

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_invoice(os.path.join(input_dir, filename), output_dir)

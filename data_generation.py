import random
import os
import csv
from faker import Faker
from reportlab.pdfgen import canvas

fake = Faker()


def generate_bank_application():
    data = {
        'name': fake.name(),
        'email': fake.email(),
        'ssn': fake.ssn(),
        'address': fake.address(),
        'phone': fake.phone_number(),
        'account_type': random.choice(['Savings', 'Checking', 'Credit Card']),
        'income': f"${random.randint(30000, 150000)}"
    }
    return "bank_application", data


def generate_id_document():
    data = {
        'name': fake.name(),
        'dob': fake.date_of_birth().strftime("%m/%d/%Y"),
        'id_number': fake.random_number(digits=8),
        'address': fake.address(),
        'expiry_date': fake.future_date().strftime("%m/%d/%Y")
    }
    return "identity_document", data


def generate_financial_document():
    data = {
        'name': fake.name(),
        'document_type': random.choice(['Income Statement', 'Tax Return', 'Paystub']),
        'year': random.randint(2018, 2023),
        'total_income': f"${random.randint(30000, 150000)}",
        'tax_paid': f"${random.randint(1000, 5000)}",
        'employer': fake.company()
    }
    return "financial_document", data


def generate_receipt():
    data = {
        'receipt_number': fake.uuid4()[:8].upper(),
        'purchase_date': fake.date_this_year().strftime("%m/%d/%Y"),
        'items': f"{random.randint(1, 5)} items",
        'total_amount': f"${random.uniform(10.0, 500.0):.2f}",
        'store': fake.company(),
        'store_address': fake.address()
    }
    return "receipt", data


def create_pdf(doc_type, data, output_path):
    try:
        c = canvas.Canvas(output_path)
        c.setFont(random.choice(["Helvetica", "Times-Roman", "Courier"]), random.randint(10, 14))

        y = 800
        for key, value in data.items():
            c.drawString(100, y, f"{key.replace('_', ' ').title()}: {value}")
            y -= 25

        c.drawString(100, 50, f"Document Type: {doc_type}")
        c.save()
    except Exception as e:
        print(f"Error creating PDF for {output_path}: {e}")


def save_metadata(metadata, output_dir):
    metadata_file = os.path.join(output_dir, "metadata.csv")
    try:
        with open(metadata_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
    except Exception as e:
        print(f"Error saving metadata: {e}")


def generate_dataset(num_samples=100, output_dir="data/"):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for i in range(num_samples):
        generators = [
            generate_bank_application,
            generate_id_document,
            generate_financial_document,
            generate_receipt
        ]
        doc_type, data = random.choice(generators)()

        filename = f"{doc_type}_{i}.pdf"
        output_path = os.path.join(output_dir, filename)
        create_pdf(doc_type, data, output_path)

        # Track metadata
        data['file_name'] = filename
        data['doc_type'] = doc_type
        metadata.append(data)

    if metadata:
        save_metadata(metadata, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic documents for classification.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of documents to generate.")
    parser.add_argument("--output_dir", type=str, default="data/", help="Directory to save generated documents.")
    args = parser.parse_args()

    generate_dataset(num_samples=args.num_samples, output_dir=args.output_dir)

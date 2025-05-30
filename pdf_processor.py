import argparse
import sqlite3
import os
import pypdf
from transformers import pipeline

# --- Configuration ---
DATABASE_FILE = "research_data.db"
SUMMARIZER_MODEL = "facebook/bart-large-cnn" # A common summarization model
# "t5-small" is faster but less accurate

# --- Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pypdf."""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None
    return text

def summarize_text(text):
    """Generates a summary of the given text using a transformers model."""
    if not text or len(text.strip()) < 50: # Avoid summarizing very short texts
        return "Text is too short to summarize."
    try:
        # Load the summarization pipeline (this might take time the first time)
        print(f"Loading summarization model: {SUMMARIZER_MODEL}...")
        summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
        print("Model loaded successfully.")

        # Adjust parameters as needed (max_length, min_length, do_sample)
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error generating summary."

def save_to_database(pdf_filename, extracted_text, summary):
    """Saves the text and summary to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_filename TEXT UNIQUE,
                extracted_text TEXT,
                summary TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert data or update if filename already exists
        cursor.execute("""
            INSERT INTO pdf_data (pdf_filename, extracted_text, summary)
            VALUES (?, ?, ?)
            ON CONFLICT(pdf_filename) DO UPDATE SET
                extracted_text = excluded.extracted_text,
                summary = excluded.summary,
                timestamp = CURRENT_TIMESTAMP
        """, (pdf_filename, extracted_text, summary))

        conn.commit()
        print(f"Data for '{pdf_filename}' saved to '{DATABASE_FILE}'.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a PDF, summarize it, and save to a database.")
    parser.add_argument("pdf_file", help="Path to the PDF file to process.")
    args = parser.parse_args()

    pdf_path = args.pdf_file

    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
    elif not pdf_path.lower().endswith(".pdf"):
        print(f"Error: File '{pdf_path}' does not appear to be a PDF.")
    else:
        print(f"Processing PDF: {pdf_path}")
        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text:
            print("Text extracted successfully. Summarizing...")
            summary = summarize_text(extracted_text)
            print("Summary generated. Saving to database...")
            # Use just the filename for database entry
            save_to_database(os.path.basename(pdf_path), extracted_text, summary)
        else:
            print("Failed to extract text. Aborting.")

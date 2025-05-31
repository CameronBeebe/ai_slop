from transformers import pipeline
import requests
import re
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import argparse
from pathlib import Path


class PDFProcessor:
    def __init__(self):
        """Initialize the PDFProcessor with a summarization pipeline."""
        self.local_model = "facebook/bart-large-cnn"
        self.summarizer = pipeline(
            "summarization",
            model=self.local_model,
            model_kwargs={"local_files_only": True}  # Only for model loading
        )

    def process_pdf(self, pdf_path):
        """
        Process a PDF file to extract text, summarize content, and generate a BibTeX entry.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            tuple: (extracted text, summary, BibTeX entry) or (None, None, error message) on failure.
        """

        # pdf_path must be path object
        pdf_path = Path(pdf_path)

        
        try:
            text = self.extract_text(pdf_path)
            metadata = self.extract_metadata(pdf_path)
            summary = self.summarize(text)
            bibtex = self.get_bibtex(metadata, text)
            return text, summary, bibtex
        except Exception as e:
            return None, None, f"Error processing PDF: {e}"

    def extract_text(self, pdf_path):
        """
        Extract text from the PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            str: Extracted text.
        """
        return extract_text(pdf_path)

    def extract_metadata(self, pdf_path):
        """
        Extract metadata from the PDF file.
        
        Args:
        pdf_path (Path): Path to the PDF file.
        
        Returns:
        dict: Metadata dictionary with string values.
        """
        with open(pdf_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            if doc.info:
                metadata = {}
                for k, v in doc.info[0].items():
                    if hasattr(v, 'resolve'):
                        # Resolve references and convert to string
                        metadata[k] = str(v.resolve())
                    elif isinstance(v, bytes):
                        # Decode bytes to string using UTF-8
                        metadata[k] = v.decode('utf-8', errors='ignore')
                    else:
                        # Convert other types to string
                        metadata[k] = str(v)
                return metadata
            return {}

    def summarize(self, text):
        """
        Summarize the extracted text using a pre-trained model.
        
        Args:
            text (str): Text to summarize.
            
        Returns:
            str: Summary of the text.
        """
        # Truncate text to avoid exceeding model limits
        max_input_length = 16384
        if len(text) > max_input_length:
            text = text[-max_input_length:]
        summary = self.summarizer(text, max_length=350, min_length=50, do_sample=False)
        return summary[0]['summary_text']

    def get_bibtex(self, metadata, text):
        """
        Generate a BibTeX entry using metadata or text.
        
        Args:
            metadata (dict): PDF metadata.
            text (str): Extracted text from the PDF.
            
        Returns:
            str: BibTeX entry or an error message.
        """
        # Try to get DOI from metadata or text
        doi = metadata.get('DOI')
        if not doi:
            doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
            match = re.search(doi_pattern, text, re.IGNORECASE)
            if match:
                doi = match.group(0)

        if doi:
            # Fetch BibTeX using DOI
            url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            else:
                return "BibTeX not found for DOI"

        # If no DOI, search by title
        title = metadata.get('Title')
        if title:
            url = f"https://api.crossref.org/works?query.title={title}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                items = data['message'].get('items', [])
                if items:
                    first_item = items[0]
                    doi = first_item.get('DOI')
                    if doi:
                        bibtex_url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
                        bibtex_response = requests.get(bibtex_url)
                        if bibtex_response.status_code == 200:
                            return bibtex_response.text
            return "BibTeX not found for title"

        return "Insufficient metadata to generate BibTeX"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF file to extract text, summarize, and generate BibTeX.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--text", action="store_true", help="Print the extracted text")
    args = parser.parse_args()

    processor = PDFProcessor()
    text, summary, bibtex_or_error = processor.process_pdf(args.pdf_path)

    if text is None:
        print(bibtex_or_error)  # Error message
    else:
        if args.text:
            print("Extracted Text:")
            print(text)
            print("\n")
        print("Summary:")
        print(summary)
        print("\n")
        print("BibTeX Entry:")
        print(bibtex_or_error)

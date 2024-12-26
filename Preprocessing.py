import pdf2image
import pytesseract
import numpy as np
import cv2
from PIL import Image
import os


class DocumentPreprocessor:
    def __init__(self, image_size=(1024, 768), tesseract_config=""):
        self.image_size = image_size
        self.tesseract_config = tesseract_config

    def preprocess_image(self, image):
        """
        Preprocesses the input image for OCR:
        - Converts to grayscale
        - Applies binary thresholding
        - Denoises the image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Denoise the image
            denoised = cv2.fastNlMeansDenoising(binary, h=30)

            return Image.fromarray(denoised)
        except Exception as e:
            print(f"Error in preprocessing image: {e}")
            return image

    def extract_text(self, image):
        """
        Extracts text and layout data from the input image using Tesseract OCR.
        """
        try:
            processed_image = self.preprocess_image(image)

            # Extract text
            text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)

            # Extract layout information
            layout = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)

            return text, layout
        except Exception as e:
            print(f"Error in text extraction: {e}")
            return "", {}

    def pdf_to_images(self, pdf_path):
        """
        Converts a multi-page PDF into images (one per page).
        """
        try:
            images = pdf2image.convert_from_path(pdf_path)
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []

    def process_pdf(self, pdf_path, output_dir="processed_text/"):
        """
        Processes a PDF file to extract text and layout information.
        Saves the extracted data to a text file.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            images = self.pdf_to_images(pdf_path)

            all_text = []
            for i, image in enumerate(images):
                text, _ = self.extract_text(image)
                all_text.append(f"Page {i + 1}:\n{text}")

            # Save the text to a file
            output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path)}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_text))

            print(f"Processed PDF saved to: {output_file}")
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

    def batch_process_pdfs(self, pdf_dir, output_dir="processed_text/"):
        """
        Processes all PDF files in a directory and saves the extracted text.
        """
        try:
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                self.process_pdf(pdf_path, output_dir)
        except Exception as e:
            print(f"Error in batch processing PDFs: {e}")
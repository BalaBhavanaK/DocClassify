# DocClassify: Automated Financial Document Processing System

## Overview
An intelligent document processing system that automatically classifies financial documents and extracts relevant information using advanced machine learning techniques. Built for Appian Credit Union, this system streamlines the processing of bank applications, identity documents, financial records, and receipts.

## 🚀 Features
- Automated document classification using LayoutLMv2
- Intelligent information extraction (personal details, IDs, etc.)
- Real-time document processing
- User-friendly Streamlit interface
- Comprehensive error handling and validation
- High accuracy and confidence scoring

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Document Processing**: PyTesseract, PDF2Image
- **ML Models**: LayoutLMv2, Spacy
- **Image Processing**: OpenCV, NumPy
- **Development**: Python 3.8+

## 📋 Prerequisites
```bash
- Python 3.8 or higher
- Tesseract OCR
- Virtual environment (recommended)
```

## 🔧 Installation
1. Clone the repository
```bash
git clone https://github.com/BalaBhavanaK/DocClassify.git
cd DocClassify
```

2. Create and activate virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 🚀 Usage
1. Start the application
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload a document and view the results

## 📁 Project Structure
```
document-intelligence/
├── src/
│   ├── data_generation/    # Synthetic data creation
│   ├── preprocessing/      # Document processing pipeline
│   ├── models/            # Classification models
│   ├── utils/             # Helper functions
│   └── app.py            # Main Streamlit application
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## 🌟 Key Capabilities
1. **Document Classification**
   - Bank account applications
   - Identity documents
   - Financial records
   - Receipts

2. **Information Extraction**
   - Personal details
   - ID numbers
   - Financial information
   - Dates and amounts

3. **Validation & Error Handling**
   - Document quality checks
   - Data validation
   - Confidence scoring

## 🔄 Processing Pipeline
1. Document Upload
2. Preprocessing & OCR
3. Document Classification
4. Information Extraction
5. Validation & Results Display

## 📊 Performance
- Classification Accuracy: ~92%
- Processing Time: < 3 seconds/document
- Information Extraction Accuracy: > 95%

## 🔜 Future Improvements
- Multi-language support
- Batch processing capabilities
- Advanced document tampering detection
- Digital signature verification
- API integration

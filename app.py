import streamlit as st
import tempfile
import os
from datetime import datetime
from preprocessing.processor import DocumentPreprocessor
from models.classifier import DocumentClassifier
from utils.info_extractor import InfoExtractor
from utils.error_handler import validate_document, ValidationError


def setup_page_config():
    st.set_page_config(
        page_title="Document Intelligence System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #3498db;
        }
        .upload-box {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)


def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def display_confidence_meter(confidence):
    if confidence >= 0.8:
        color = "green"
    elif confidence >= 0.6:
        color = "yellow"
    else:
        color = "red"

    st.markdown(f"""
        <div style="border:1px solid {color}; padding:10px; border-radius:5px">
            <div style="width:{confidence * 100}%; background-color:{color}; height:20px; border-radius:3px"></div>
        </div>
        <p style="text-align:center">{confidence:.1%}</p>
    """, unsafe_allow_html=True)


def main():
    setup_page_config()

    # Sidebar
    with st.sidebar:
        st.image("path_to_logo.png", width=200)  # Add your logo
        st.markdown("## Document Types")
        st.markdown("• Bank Applications\n• Identity Documents\n• Financial Records\n• Receipts")

        st.markdown("---")
        st.markdown("### Processing Status")
        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = 0
        st.metric("Documents Processed", st.session_state.processed_docs)

    # Main content
    st.title("📄 Document Intelligence System")
    st.markdown("### Upload and Process Financial Documents")

    # Initialize components
    processor = DocumentPreprocessor()
    classifier = DocumentClassifier()
    info_extractor = InfoExtractor()

    # File upload with progress
    uploaded_file = st.file_uploader(
        "Drag and drop your document here",
        type=['pdf'],
        help="Supported format: PDF"
    )

    if uploaded_file:
        temp_path = save_uploaded_file(uploaded_file)
        if not temp_path:
            return

        try:
            with st.spinner('Processing your document...'):
                # Progress bar for processing steps
                progress_bar = st.progress(0)

                # Step 1: Convert to image
                image = processor.pdf_to_image(temp_path)
                progress_bar.progress(25)

                # Step 2: Extract text and layout
                text, layout = processor.extract_text(image)
                progress_bar.progress(50)

                # Step 3: Classify document
                classification = classifier.predict(image, text, layout)
                progress_bar.progress(75)

                # Step 4: Extract information
                extracted_info = info_extractor.extract(text)
                progress_bar.progress(100)

                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📑 Overview", "👤 Details", "📝 Raw Text"])

                with tab1:
                    st.subheader("Document Analysis Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Document Type")
                        st.info(classification['document_type'].replace('_', ' ').title())

                        st.markdown("#### Confidence Score")
                        display_confidence_meter(classification['confidence'])

                    with col2:
                        st.markdown("#### Document Preview")
                        st.image(image, caption='Uploaded Document', use_column_width=True)

                with tab2:
                    st.markdown("#### Extracted Information")
                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        st.markdown("##### Personal Details")
                        st.write(f"**Name:** {extracted_info.get('name', 'Not Found')}")
                        st.write(f"**Email:** {extracted_info.get('email', 'Not Found')}")
                        st.write(f"**ID:** {extracted_info.get('id_number', 'Not Found')}")

                    with info_col2:
                        st.markdown("##### Additional Information")
                        st.write(f"**Address:** {extracted_info.get('address', 'Not Found')}")
                        st.write(f"**Phone:** {extracted_info.get('phone', 'Not Found')}")
                        st.write(f"**Date:** {extracted_info.get('date', ['Not Found'])[0]}")

                with tab3:
                    st.markdown("#### Extracted Text")
                    st.text_area("Full Document Text", text, height=300)

                # Update processed documents count
                st.session_state.processed_docs += 1

                # Validation results
                try:
                    validate_document(classification['document_type'], extracted_info)
                    st.success("✅ Document validation passed")
                except ValidationError as e:
                    st.warning(f"⚠️ Validation notice: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    main()
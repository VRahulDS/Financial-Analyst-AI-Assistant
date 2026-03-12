import streamlit as st
from paths import RAWFILES_DIR
import logging
logger = logging.getLogger(__name__)

def upload_document():
    uploaded_file = st.file_uploader("Upload financial report", type=["pdf"])
    logger.info("uploading the pdf file")
    try:
        if uploaded_file:
            file_path = RAWFILES_DIR / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved to {file_path}")
            logger.info("uploaded the pdf file successfully")
            return file_path
    except Exception as e:
        logger.error(f"Failed to upload the pdf: {e}")
"""
Streamlit web interface for handwriting recognition.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional
import tempfile
import os

from handwriting_recognizer import HandwritingRecognizer, create_sample_image
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-info {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

def load_model():
    """Load the handwriting recognition model."""
    if not st.session_state.model_loaded:
        try:
            with st.spinner("Loading TrOCR model..."):
                st.session_state.recognizer = HandwritingRecognizer()
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.session_state.model_loaded = False

def process_image(uploaded_file) -> Optional[str]:
    """Process uploaded image and return recognized text."""
    if st.session_state.recognizer is None:
        st.error("Model not loaded. Please load the model first.")
        return None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process the image
        recognized_text = st.session_state.recognizer.recognize_from_file(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return recognized_text
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">✍️ Handwriting Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model loading section
        st.subheader("Model")
        if st.button("Load Model", type="primary"):
            load_model()
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
            
            # Display model info
            model_info = st.session_state.recognizer.get_model_info()
            st.markdown("### Model Information")
            st.markdown(f"""
            <div class="model-info">
            <strong>Model:</strong> {model_info['model_name']}<br>
            <strong>Device:</strong> {model_info['device']}<br>
            <strong>Status:</strong> Loaded ✅
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model Not Loaded")
        
        # Sample image section
        st.subheader("Sample Data")
        if st.button("Generate Sample Image"):
            sample_image = create_sample_image()
            st.image(sample_image, caption="Sample Handwriting", use_column_width=True)
            
            # Save sample image
            sample_path = Path("data/sample_handwriting.png")
            sample_path.parent.mkdir(exist_ok=True)
            sample_image.save(sample_path)
            st.success("Sample image saved to data/sample_handwriting.png")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing handwritten text"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Recognize Text", type="primary", disabled=not st.session_state.model_loaded):
                if st.session_state.model_loaded:
                    with st.spinner("Processing image..."):
                        recognized_text = process_image(uploaded_file)
                        if recognized_text:
                            st.session_state.last_result = recognized_text
                else:
                    st.error("Please load the model first.")
    
    with col2:
        st.header("Recognition Result")
        
        if st.session_state.last_result:
            st.markdown(f"""
            <div class="result-box">
            <h3>Recognized Text:</h3>
            <p style="font-size: 1.2rem; font-weight: bold;">{st.session_state.last_result}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Copy to clipboard button
            st.code(st.session_state.last_result, language=None)
            
            # Download result
            st.download_button(
                label="Download Result",
                data=st.session_state.last_result,
                file_name="recognized_text.txt",
                mime="text/plain"
            )
        else:
            st.info("Upload an image and click 'Recognize Text' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    Handwriting Recognition using TrOCR (Transformer OCR) by Microsoft<br>
    Built with Streamlit and Hugging Face Transformers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

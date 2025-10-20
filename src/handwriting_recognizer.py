"""
Handwriting Recognition Module

This module provides functionality for recognizing handwritten text using the TrOCR model
from Microsoft's transformers library. It includes error handling, type hints, and
modern Python practices.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HandwritingRecognizer:
    """
    A class for recognizing handwritten text using TrOCR model.
    
    This class encapsulates the TrOCR model and processor for handwriting recognition,
    providing a clean interface for text extraction from handwritten images.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten") -> None:
        """
        Initialize the HandwritingRecognizer.
        
        Args:
            model_name: The name of the TrOCR model to use. Defaults to the handwritten variant.
        """
        self.model_name = model_name
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the TrOCR processor and model."""
        try:
            logger.info(f"Loading TrOCR model: {self.model_name}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and preprocess an image for handwriting recognition.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            PIL Image object converted to RGB.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be opened.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Successfully loaded image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise ValueError(f"Cannot open image: {e}")
    
    def recognize_text(self, image: Image.Image) -> str:
        """
        Recognize handwritten text from an image.
        
        Args:
            image: PIL Image object containing handwritten text.
            
        Returns:
            Recognized text as a string.
            
        Raises:
            RuntimeError: If the model is not loaded or recognition fails.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not loaded. Please initialize the recognizer.")
        
        try:
            # Preprocess the image
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            
            # Generate text prediction
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode the predicted text
            recognized_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            logger.info(f"Successfully recognized text: '{recognized_text}'")
            return recognized_text
            
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            raise RuntimeError(f"Recognition failed: {e}")
    
    def recognize_from_file(self, image_path: Union[str, Path]) -> str:
        """
        Recognize handwritten text from an image file.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Recognized text as a string.
        """
        image = self.load_image(image_path)
        return self.recognize_text(image)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information.
        """
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "device": str(next(self.model.parameters()).device) if self.model else None
        }


def create_sample_image() -> Image.Image:
    """
    Create a sample handwritten image for testing purposes.
    
    Returns:
        PIL Image object with sample handwritten text.
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white background image
    width, height = 400, 100
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a handwriting-like font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Draw sample text
    text = "Hello World!"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    
    return image


if __name__ == "__main__":
    # Example usage
    recognizer = HandwritingRecognizer()
    
    # Create and save a sample image
    sample_image = create_sample_image()
    sample_path = Path("data/sample_handwriting.png")
    sample_path.parent.mkdir(exist_ok=True)
    sample_image.save(sample_path)
    
    # Recognize text from the sample image
    try:
        recognized_text = recognizer.recognize_from_file(sample_path)
        print(f"Recognized text: {recognized_text}")
        print(f"Model info: {recognizer.get_model_info()}")
    except Exception as e:
        print(f"Error: {e}")

"""
Handwriting Recognition Package

A modern, production-ready handwriting recognition system using Microsoft's TrOCR model.
"""

from .handwriting_recognizer import HandwritingRecognizer, create_sample_image
from .config import Config, config
from .visualization import HandwritingVisualizer, create_sample_dataset

__version__ = "1.0.0"
__author__ = "Handwriting Recognition Team"
__email__ = "team@handwriting-recognition.com"

__all__ = [
    "HandwritingRecognizer",
    "create_sample_image", 
    "Config",
    "config",
    "HandwritingVisualizer",
    "create_sample_dataset"
]

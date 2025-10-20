"""
Unit tests for handwriting recognition module.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from src.handwriting_recognizer import HandwritingRecognizer, create_sample_image
from src.config import Config


class TestHandwritingRecognizer(unittest.TestCase):
    """Test cases for HandwritingRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = create_sample_image()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.handwriting_recognizer.TrOCRProcessor')
    @patch('src.handwriting_recognizer.VisionEncoderDecoderModel')
    def test_initialization(self, mock_model_class, mock_processor_class):
        """Test HandwritingRecognizer initialization."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        recognizer = HandwritingRecognizer()
        
        self.assertEqual(recognizer.model_name, "microsoft/trocr-base-handwritten")
        self.assertIsNotNone(recognizer.processor)
        self.assertIsNotNone(recognizer.model)
    
    def test_load_image_valid_path(self):
        """Test loading a valid image."""
        # Save test image to temporary file
        temp_path = Path(self.temp_dir) / "test_image.png"
        self.test_image.save(temp_path)
        
        recognizer = HandwritingRecognizer.__new__(HandwritingRecognizer)
        
        loaded_image = recognizer.load_image(temp_path)
        self.assertIsInstance(loaded_image, Image.Image)
        self.assertEqual(loaded_image.mode, 'RGB')
    
    def test_load_image_invalid_path(self):
        """Test loading an image from invalid path."""
        recognizer = HandwritingRecognizer.__new__(HandwritingRecognizer)
        
        with self.assertRaises(FileNotFoundError):
            recognizer.load_image("nonexistent_image.png")
    
    @patch('src.handwriting_recognizer.TrOCRProcessor')
    @patch('src.handwriting_recognizer.VisionEncoderDecoderModel')
    def test_recognize_text(self, mock_model_class, mock_processor_class):
        """Test text recognition from image."""
        # Mock the processor and model
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock the processing pipeline
        mock_processor.return_value.pixel_values = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4]]  # Mock token IDs
        mock_processor.batch_decode.return_value = ["Hello World"]
        
        recognizer = HandwritingRecognizer()
        result = recognizer.recognize_text(self.test_image)
        
        self.assertEqual(result, "Hello World")
        mock_processor.assert_called_once()
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_once()
    
    def test_recognize_text_no_model(self):
        """Test text recognition when model is not loaded."""
        recognizer = HandwritingRecognizer.__new__(HandwritingRecognizer)
        recognizer.processor = None
        recognizer.model = None
        
        with self.assertRaises(RuntimeError):
            recognizer.recognize_text(self.test_image)
    
    def test_get_model_info(self):
        """Test getting model information."""
        recognizer = HandwritingRecognizer.__new__(HandwritingRecognizer)
        recognizer.model_name = "test-model"
        recognizer.processor = MagicMock()
        recognizer.model = MagicMock()
        
        # Mock model parameters
        mock_param = MagicMock()
        mock_param.device = "cpu"
        recognizer.model.parameters.return_value = [mock_param]
        
        info = recognizer.get_model_info()
        
        self.assertEqual(info["model_name"], "test-model")
        self.assertTrue(info["model_loaded"])
        self.assertTrue(info["processor_loaded"])
        self.assertEqual(info["device"], "cpu")


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config(self.config_path)
        
        self.assertEqual(config.get("model.name"), "microsoft/trocr-base-handwritten")
        self.assertEqual(config.get("logging.level"), "INFO")
        self.assertEqual(config.get("web_app.port"), 8501)
    
    def test_set_and_get_config(self):
        """Test setting and getting configuration values."""
        config = Config(self.config_path)
        
        config.set("test.value", "test_data")
        self.assertEqual(config.get("test.value"), "test_data")
        
        config.set("model.name", "custom-model")
        self.assertEqual(config.get("model.name"), "custom-model")
    
    def test_get_with_default(self):
        """Test getting configuration with default value."""
        config = Config(self.config_path)
        
        value = config.get("nonexistent.key", "default_value")
        self.assertEqual(value, "default_value")
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config(self.config_path)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn("model", config_dict)
        self.assertIn("logging", config_dict)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_sample_image(self):
        """Test creating a sample image."""
        image = create_sample_image()
        
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, 'RGB')
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)


if __name__ == '__main__':
    # Run tests
    unittest.main()

# Handwriting Recognition Project

A production-ready handwriting recognition system using Microsoft's TrOCR (Transformer OCR) model. This project provides both a Python API and a web interface for recognizing handwritten text from images.

## Features

- **State-of-the-art Model**: Uses Microsoft's TrOCR model trained on handwritten text
- **High Accuracy**: Optimized for handwritten text recognition
- **Web Interface**: User-friendly Streamlit web application
- **Configurable**: YAML-based configuration management
- **Tested**: Comprehensive unit tests
- **Logging**: Detailed logging for debugging and monitoring
- **Type Hints**: Full type annotation support
- **Modern Python**: Follows PEP8 and modern Python practices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Handwriting-Recognition-Project.git
   cd Handwriting-Recognition-Project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Using the Python API

```python
from src.handwriting_recognizer import HandwritingRecognizer

# Initialize the recognizer
recognizer = HandwritingRecognizer()

# Recognize text from an image file
recognized_text = recognizer.recognize_from_file("path/to/your/image.jpg")
print(f"Recognized text: {recognized_text}")
```

### Using the Web Interface

1. **Start the Streamlit app**:
   ```bash
   streamlit run web_app/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload an image** containing handwritten text and click "Recognize Text"

## Project Structure

```
handwriting-recognition/
├── src/                          # Source code
│   ├── handwriting_recognizer.py # Main recognition module
│   └── config.py                 # Configuration management
├── web_app/                      # Web interface
│   └── app.py                   # Streamlit application
├── tests/                        # Unit tests
│   └── test_handwriting_recognizer.py
├── data/                         # Data directory
├── models/                       # Model storage
├── config/                       # Configuration files
│   └── config.yaml              # Default configuration
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Configuration

The project uses YAML configuration files. The default configuration is located in `config/config.yaml`:

```yaml
model:
  name: "microsoft/trocr-base-handwritten"
  device: "auto"  # auto, cpu, cuda
  max_length: 256

paths:
  data_dir: "data"
  models_dir: "models"
  output_dir: "output"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

web_app:
  host: "0.0.0.0"
  port: 8501
  title: "Handwriting Recognition"
```

## API Reference

### HandwritingRecognizer Class

#### `__init__(model_name: str = "microsoft/trocr-base-handwritten")`
Initialize the handwriting recognizer with a specific TrOCR model.

#### `load_image(image_path: Union[str, Path]) -> Image.Image`
Load and preprocess an image for recognition.

#### `recognize_text(image: Image.Image) -> str`
Recognize handwritten text from a PIL Image object.

#### `recognize_from_file(image_path: Union[str, Path]) -> str`
Recognize handwritten text from an image file path.

#### `get_model_info() -> dict`
Get information about the loaded model.

## Examples

### Basic Usage

```python
from src.handwriting_recognizer import HandwritingRecognizer

# Create recognizer instance
recognizer = HandwritingRecognizer()

# Process an image
text = recognizer.recognize_from_file("handwriting_sample.jpg")
print(f"Recognized: {text}")
```

### Batch Processing

```python
from pathlib import Path
from src.handwriting_recognizer import HandwritingRecognizer

recognizer = HandwritingRecognizer()
image_dir = Path("images")

for image_path in image_dir.glob("*.jpg"):
    try:
        text = recognizer.recognize_from_file(image_path)
        print(f"{image_path.name}: {text}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
```

### Custom Configuration

```python
from src.config import Config

# Load custom configuration
config = Config("custom_config.yaml")

# Update settings
config.set("model.device", "cuda")
config.set("logging.level", "DEBUG")
config.save()
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src

# Run specific test file
python -m pytest tests/test_handwriting_recognizer.py
```

## Development

### Code Style

The project follows PEP8 style guidelines. Format code using:

```bash
black src/ tests/ web_app/
flake8 src/ tests/ web_app/
```

### Type Checking

Run type checking with mypy:

```bash
mypy src/
```

## Performance Tips

1. **GPU Acceleration**: Set `device: "cuda"` in configuration if you have a compatible GPU
2. **Batch Processing**: Process multiple images in batches for better throughput
3. **Image Preprocessing**: Ensure images are properly cropped and oriented
4. **Model Caching**: The model is loaded once and reused for multiple predictions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check internet connection and model availability
3. **Image Format Issues**: Ensure images are in supported formats (PNG, JPG, JPEG)

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft Research for the TrOCR model
- Hugging Face for the transformers library
- Streamlit for the web interface framework

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.
# Handwriting-Recognition-Project

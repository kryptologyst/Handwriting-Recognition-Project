"""
Command-line interface for handwriting recognition.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.handwriting_recognizer import HandwritingRecognizer, create_sample_image
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Handwriting Recognition CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recognize text from a single image
  python cli.py recognize image.jpg

  # Recognize text from multiple images
  python cli.py recognize *.jpg

  # Generate a sample image
  python cli.py generate-sample

  # Batch process images in a directory
  python cli.py batch-process images/ --output results.txt

  # Start web interface
  python cli.py web
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Recognize command
    recognize_parser = subparsers.add_parser('recognize', help='Recognize text from image(s)')
    recognize_parser.add_argument('images', nargs='+', help='Image file(s) to process')
    recognize_parser.add_argument('--output', '-o', help='Output file for results')
    recognize_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Generate sample command
    sample_parser = subparsers.add_parser('generate-sample', help='Generate a sample handwritten image')
    sample_parser.add_argument('--output', '-o', default='sample_handwriting.png', help='Output filename')
    
    # Batch process command
    batch_parser = subparsers.add_parser('batch-process', help='Process multiple images in batch')
    batch_parser.add_argument('input_dir', help='Input directory containing images')
    batch_parser.add_argument('--output', '-o', help='Output file for results')
    batch_parser.add_argument('--pattern', default='*.jpg', help='File pattern to match')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=8501, help='Port to bind to')
    
    # Model info command
    info_parser = subparsers.add_parser('model-info', help='Show model information')
    
    return parser


def recognize_images(image_paths: List[str], output_file: Optional[str] = None, verbose: bool = False) -> None:
    """Recognize text from multiple images."""
    recognizer = HandwritingRecognizer()
    
    results = []
    
    for image_path in image_paths:
        try:
            if verbose:
                logger.info(f"Processing {image_path}")
            
            recognized_text = recognizer.recognize_from_file(image_path)
            result = f"{image_path}: {recognized_text}"
            results.append(result)
            
            if verbose:
                print(f"✅ {result}")
            else:
                print(recognized_text)
                
        except Exception as e:
            error_msg = f"{image_path}: ERROR - {e}"
            results.append(error_msg)
            logger.error(f"Failed to process {image_path}: {e}")
            print(f"❌ {error_msg}")
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        logger.info(f"Results saved to {output_file}")


def generate_sample(output_path: str) -> None:
    """Generate a sample handwritten image."""
    try:
        sample_image = create_sample_image()
        sample_image.save(output_path)
        print(f"✅ Sample image generated: {output_path}")
        logger.info(f"Sample image saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to generate sample image: {e}")
        logger.error(f"Sample generation failed: {e}")


def batch_process(input_dir: str, output_file: Optional[str], pattern: str) -> None:
    """Process all images in a directory."""
    input_path = Path(input_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Find all matching image files
    image_files = list(input_path.glob(pattern))
    
    if not image_files:
        print(f"❌ No images found matching pattern '{pattern}' in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    image_paths = [str(f) for f in image_files]
    recognize_images(image_paths, output_file, verbose=True)


def start_web_interface(host: str, port: int) -> None:
    """Start the web interface."""
    try:
        import subprocess
        import sys
        
        cmd = [sys.executable, "-m", "streamlit", "run", "web_app/app.py", "--server.address", host, "--server.port", str(port)]
        print(f"🚀 Starting web interface at http://{host}:{port}")
        subprocess.run(cmd)
    except ImportError:
        print("❌ Streamlit not installed. Please install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        logger.error(f"Web interface startup failed: {e}")


def show_model_info() -> None:
    """Show model information."""
    try:
        recognizer = HandwritingRecognizer()
        info = recognizer.get_model_info()
        
        print("📊 Model Information:")
        print(f"  Model Name: {info['model_name']}")
        print(f"  Model Loaded: {'✅' if info['model_loaded'] else '❌'}")
        print(f"  Processor Loaded: {'✅' if info['processor_loaded'] else '❌'}")
        print(f"  Device: {info['device']}")
        
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")
        logger.error(f"Model info retrieval failed: {e}")


def main():
    """Main CLI function."""
    parser = setup_args()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'recognize':
            recognize_images(args.images, args.output, args.verbose)
        elif args.command == 'generate-sample':
            generate_sample(args.output)
        elif args.command == 'batch-process':
            batch_process(args.input_dir, args.output, args.pattern)
        elif args.command == 'web':
            start_web_interface(args.host, args.port)
        elif args.command == 'model-info':
            show_model_info()
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

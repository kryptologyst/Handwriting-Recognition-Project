#!/usr/bin/env python3
"""
Demo script for handwriting recognition project.
This script demonstrates the main functionality of the handwriting recognition system.
"""

import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from handwriting_recognizer import HandwritingRecognizer, create_sample_image
from visualization import HandwritingVisualizer, create_sample_dataset
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    print("🚀 Handwriting Recognition Demo")
    print("=" * 50)
    
    try:
        # Initialize recognizer
        print("📦 Initializing handwriting recognizer...")
        recognizer = HandwritingRecognizer()
        print("✅ Recognizer initialized successfully!")
        
        # Show model info
        model_info = recognizer.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Status: {'✅ Loaded' if model_info['model_loaded'] else '❌ Not loaded'}")
        
        # Create sample dataset
        print(f"\n🎨 Creating sample dataset...")
        sample_paths = create_sample_dataset(3, "data/demo_samples")
        print(f"✅ Created {len(sample_paths)} sample images")
        
        # Process samples
        print(f"\n🔍 Processing samples...")
        recognized_texts = []
        
        for i, path in enumerate(sample_paths, 1):
            try:
                text = recognizer.recognize_from_file(path)
                recognized_texts.append(text)
                print(f"   Sample {i}: '{text}'")
            except Exception as e:
                print(f"   Sample {i}: Error - {e}")
                recognized_texts.append("ERROR")
        
        # Create visualizations
        print(f"\n📈 Creating visualizations...")
        visualizer = HandwritingVisualizer(recognizer)
        
        # Create comparison plot
        try:
            visualizer.create_comparison_plot(
                sample_paths, 
                recognized_texts, 
                "output/demo_comparison.png"
            )
            print("✅ Comparison plot created")
        except Exception as e:
            print(f"⚠️  Comparison plot failed: {e}")
        
        # Create pipeline diagram
        try:
            visualizer.create_processing_pipeline_diagram("output/demo_pipeline.png")
            print("✅ Pipeline diagram created")
        except Exception as e:
            print(f"⚠️  Pipeline diagram failed: {e}")
        
        # Test single image recognition
        print(f"\n🖼️  Testing single image recognition...")
        if sample_paths:
            sample_image = create_sample_image()
            sample_path = Path("data/demo_single.png")
            sample_path.parent.mkdir(exist_ok=True)
            sample_image.save(sample_path)
            
            try:
                result = recognizer.recognize_from_file(sample_path)
                print(f"✅ Single image result: '{result}'")
                
                # Create single image visualization
                visualizer.visualize_recognition_result(
                    str(sample_path), 
                    result, 
                    "output/demo_single.png"
                )
                print("✅ Single image visualization created")
            except Exception as e:
                print(f"❌ Single image recognition failed: {e}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n📁 Output files created in 'output/' directory")
        print(f"📁 Sample images created in 'data/' directory")
        
        # Show configuration
        print(f"\n⚙️  Current Configuration:")
        print(f"   Data directory: {config.get('paths.data_dir')}")
        print(f"   Output directory: {config.get('paths.output_dir')}")
        print(f"   Logging level: {config.get('logging.level')}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

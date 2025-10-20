"""
Visualization and explainability module for handwriting recognition.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

from src.handwriting_recognizer import HandwritingRecognizer

logger = logging.getLogger(__name__)


class HandwritingVisualizer:
    """Visualization tools for handwriting recognition results."""
    
    def __init__(self, recognizer: Optional[HandwritingRecognizer] = None):
        """
        Initialize the visualizer.
        
        Args:
            recognizer: HandwritingRecognizer instance. If None, creates a new one.
        """
        self.recognizer = recognizer or HandwritingRecognizer()
        self.setup_style()
    
    def setup_style(self) -> None:
        """Set up matplotlib and seaborn styles."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def visualize_recognition_result(
        self, 
        image_path: str, 
        recognized_text: str,
        save_path: Optional[str] = None,
        show_confidence: bool = False
    ) -> None:
        """
        Visualize the recognition result with the original image and recognized text.
        
        Args:
            image_path: Path to the original image
            recognized_text: The recognized text
            save_path: Path to save the visualization
            show_confidence: Whether to show confidence scores (if available)
        """
        try:
            # Load the original image
            image = Image.open(image_path)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Show original image
            ax1.imshow(image)
            ax1.set_title("Original Handwritten Image", fontweight='bold')
            ax1.axis('off')
            
            # Show recognized text
            ax2.text(0.1, 0.5, recognized_text, 
                    fontsize=16, fontweight='bold',
                    verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax2.set_title("Recognized Text", fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            
            # Add overall title
            fig.suptitle("Handwriting Recognition Result", fontsize=18, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise
    
    def create_comparison_plot(
        self, 
        image_paths: List[str], 
        recognized_texts: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comparison plot showing multiple images and their recognized texts.
        
        Args:
            image_paths: List of image paths
            recognized_texts: List of recognized texts
            save_path: Path to save the plot
        """
        if len(image_paths) != len(recognized_texts):
            raise ValueError("Number of images and texts must match")
        
        n_images = len(image_paths)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (image_path, text) in enumerate(zip(image_paths, recognized_texts)):
            try:
                image = Image.open(image_path)
                axes[i].imshow(image)
                axes[i].set_title(f"Recognized: {text}", fontsize=10, fontweight='bold')
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{image_path}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"Error: {str(e)[:30]}...", fontsize=10)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle("Handwriting Recognition Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_accuracy_report(
        self, 
        ground_truth: List[str], 
        predictions: List[str],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an accuracy report comparing ground truth with predictions.
        
        Args:
            ground_truth: List of ground truth texts
            predictions: List of predicted texts
            save_path: Path to save the report plot
            
        Returns:
            Dictionary containing accuracy metrics
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("Ground truth and predictions must have the same length")
        
        # Calculate metrics
        exact_matches = sum(1 for gt, pred in zip(ground_truth, predictions) if gt.lower() == pred.lower())
        accuracy = exact_matches / len(ground_truth)
        
        # Character-level accuracy
        total_chars = sum(len(gt) for gt in ground_truth)
        correct_chars = 0
        for gt, pred in zip(ground_truth, predictions):
            min_len = min(len(gt), len(pred))
            for i in range(min_len):
                if gt[i].lower() == pred[i].lower():
                    correct_chars += 1
        
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy metrics bar chart
        metrics = ['Exact Match', 'Character-level']
        accuracies = [accuracy, char_accuracy]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(metrics, accuracies, color=colors)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Recognition Accuracy Metrics')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Sample comparison
        sample_size = min(5, len(ground_truth))
        sample_indices = np.random.choice(len(ground_truth), sample_size, replace=False)
        
        comparison_data = []
        for i in sample_indices:
            comparison_data.append({
                'Ground Truth': ground_truth[i][:30] + '...' if len(ground_truth[i]) > 30 else ground_truth[i],
                'Prediction': predictions[i][:30] + '...' if len(predictions[i]) > 30 else predictions[i],
                'Match': '✓' if ground_truth[i].lower() == predictions[i].lower() else '✗'
            })
        
        # Create comparison table
        ax2.axis('tight')
        ax2.axis('off')
        
        table_data = []
        headers = ['Ground Truth', 'Prediction', 'Match']
        for row in comparison_data:
            table_data.append([row['Ground Truth'], row['Prediction'], row['Match']])
        
        table = ax2.table(cellText=table_data, colLabels=headers, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax2.set_title(f'Sample Comparison (n={sample_size})')
        
        fig.suptitle('Handwriting Recognition Accuracy Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Accuracy report saved to {save_path}")
        
        plt.show()
        
        # Return metrics
        metrics_dict = {
            'exact_match_accuracy': accuracy,
            'character_accuracy': char_accuracy,
            'total_samples': len(ground_truth),
            'exact_matches': exact_matches,
            'total_characters': total_chars,
            'correct_characters': correct_chars
        }
        
        return metrics_dict
    
    def create_word_cloud(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """
        Create a word cloud from recognized texts.
        
        Args:
            texts: List of recognized texts
            save_path: Path to save the word cloud
        """
        try:
            from wordcloud import WordCloud
            
            # Combine all texts
            combined_text = ' '.join(texts)
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(combined_text)
            
            # Plot
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud from Recognized Texts', fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Word cloud saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("WordCloud library not installed. Install with: pip install wordcloud")
            print("WordCloud library not available. Install with: pip install wordcloud")
        except Exception as e:
            logger.error(f"Word cloud creation failed: {e}")
            raise
    
    def create_processing_pipeline_diagram(self, save_path: Optional[str] = None) -> None:
        """
        Create a diagram showing the processing pipeline.
        
        Args:
            save_path: Path to save the diagram
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define pipeline steps
        steps = [
            "Input Image",
            "Preprocessing\n(RGB conversion)",
            "TrOCR Processor\n(Image → Tokens)",
            "Vision Encoder\n(ViT)",
            "Text Decoder\n(Transformer)",
            "Output Text"
        ]
        
        # Create flowchart
        box_width = 1.5
        box_height = 0.8
        spacing = 0.3
        
        for i, step in enumerate(steps):
            x = i * (box_width + spacing)
            y = 0
            
            # Draw box
            rect = patches.Rectangle((x, y), box_width, box_height, 
                                  linewidth=2, edgecolor='navy', facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            
            # Add text
            ax.text(x + box_width/2, y + box_height/2, step, 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add arrow (except for last step)
            if i < len(steps) - 1:
                arrow_x = x + box_width + spacing/2
                arrow_y = y + box_height/2
                ax.annotate('', xy=(arrow_x + spacing/2, arrow_y), 
                           xytext=(arrow_x - spacing/2, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='navy'))
        
        ax.set_xlim(-0.2, len(steps) * (box_width + spacing) - 0.2)
        ax.set_ylim(-0.2, box_height + 0.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('Handwriting Recognition Pipeline', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pipeline diagram saved to {save_path}")
        
        plt.show()


def create_sample_dataset(n_samples: int = 10, output_dir: str = "data/samples") -> List[str]:
    """
    Create a sample dataset for testing and demonstration.
    
    Args:
        n_samples: Number of sample images to create
        output_dir: Directory to save sample images
        
    Returns:
        List of created image paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample texts
    sample_texts = [
        "Hello World!",
        "Machine Learning",
        "Deep Learning",
        "Computer Vision",
        "Natural Language Processing",
        "Artificial Intelligence",
        "Neural Networks",
        "Data Science",
        "Python Programming",
        "Handwriting Recognition"
    ]
    
    created_paths = []
    
    for i in range(min(n_samples, len(sample_texts))):
        # Create image
        image = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(image)
        
        # Try to use a handwriting-like font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        text = sample_texts[i]
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (400 - text_width) // 2
        y = (100 - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        # Save image
        image_path = output_path / f"sample_{i+1:02d}.png"
        image.save(image_path)
        created_paths.append(str(image_path))
        
        logger.info(f"Created sample image: {image_path}")
    
    return created_paths


if __name__ == "__main__":
    # Example usage
    visualizer = HandwritingVisualizer()
    
    # Create sample dataset
    sample_paths = create_sample_dataset(5)
    
    # Process samples
    recognizer = HandwritingRecognizer()
    recognized_texts = []
    
    for path in sample_paths:
        try:
            text = recognizer.recognize_from_file(path)
            recognized_texts.append(text)
            print(f"{path}: {text}")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            recognized_texts.append("ERROR")
    
    # Create visualizations
    if len(sample_paths) > 0:
        visualizer.create_comparison_plot(sample_paths, recognized_texts, "output/comparison.png")
    
    # Create pipeline diagram
    visualizer.create_processing_pipeline_diagram("output/pipeline.png")

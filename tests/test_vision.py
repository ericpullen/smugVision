#!/usr/bin/env python3
"""Simple test script for vision model integration.

This script demonstrates how to use the vision module to process images
with Ollama's llama3.2-vision model.

Usage:
    python test_vision.py <image_path>
"""

import sys
import logging
import time
from pathlib import Path

from smugvision.vision.factory import VisionModelFactory
from smugvision.vision.exceptions import VisionModelError

# Configure logging
# Use DEBUG level to see detailed face recognition logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Timing tracker
class TimingTracker:
    """Track timing for different stages of processing."""
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, stage_name):
        """Start timing a stage."""
        self.start_times[stage_name] = time.time()
    
    def end(self, stage_name):
        """End timing a stage."""
        if stage_name in self.start_times:
            elapsed = time.time() - self.start_times[stage_name]
            self.timings[stage_name] = elapsed
            del self.start_times[stage_name]
            return elapsed
        return None
    
    def print_summary(self):
        """Print timing summary."""
        print("\n" + "=" * 60)
        print("⏱️  TIMING BREAKDOWN")
        print("=" * 60)
        total = sum(self.timings.values())
        for stage, duration in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total * 100) if total > 0 else 0
            print(f"{stage:.<45} {duration:>7.2f}s ({percentage:>5.1f}%)")
        print("-" * 60)
        print(f"{'TOTAL':.<45} {total:>7.2f}s")
        print("=" * 60)


def main():
    """Test vision model with a local image."""
    if len(sys.argv) < 2:
        print("Usage: python test_vision.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Initialize timing tracker
    timer = TimingTracker()
    
    # Default prompts (can be customized)
    caption_prompt = (
        "Analyze this image and provide a concise, descriptive caption "
        "(1-2 sentences) that describes the main subject, setting, and "
        "any notable activities or features."
    )
    
    tags_prompt = (
        "Generate 5-10 simple, single-word or short-phrase keyword tags for this image. "
        "Use simple descriptive words like: restaurant, eating, family, indoor, etc. "
        "Focus on:\n"
        "- Main subjects and objects (one word each)\n"
        "- Activities or actions (one word each)\n"
        "- Setting and location (one word each)\n"
        "- Colors and mood (one word each)\n"
        "- Time of day or season (if apparent)\n"
        "Provide ONLY a comma-separated list of simple tags, nothing else. "
        "Example format: restaurant, eating, family, indoor, casual, warm"
    )
    
    try:
        # Create vision model using factory
        print(f"Initializing Llama 3.2 Vision model...")
        timer.start("1. Model Initialization")
        model = VisionModelFactory.create(
            model_name="llama3.2-vision",
            endpoint="http://localhost:11434"
        )
        timer.end("1. Model Initialization")
        
        print(f"\nProcessing image: {image_path}")
        print("-" * 60)
        
        # Optional: Initialize face recognizer if reference faces directory exists
        face_recognizer = None
        reference_faces_dir = Path.home() / ".smugvision" / "reference_faces"
        if reference_faces_dir.exists():
            try:
                from smugvision.face import FaceRecognizer
                print(f"\nLoading reference faces from: {reference_faces_dir}")
                timer.start("2. Face Recognizer Initialization")
                face_recognizer = FaceRecognizer(str(reference_faces_dir))
                timer.end("2. Face Recognizer Initialization")
                print(f"  Loaded {len(face_recognizer.reference_faces)} person(s)")
            except ImportError as e:
                print(f"  Face recognition not available: {e}")
                print("  Install with:")
                print("    pip install face_recognition")
                print("    pip install git+https://github.com/ageitgey/face_recognition_models")
            except Exception as e:
                print(f"  Could not load face recognizer: {e}")
        
        # Process image (EXIF location and face recognition will be automatically used)
        timer.start("3. Total Image Processing")
        result = model.process_image(
            image_path=str(image_path),
            caption_prompt=caption_prompt,
            tags_prompt=tags_prompt,
            temperature=0.7,
            max_tokens=150,
            generate_caption=True,
            generate_tags=True,
            use_exif_location=True,  # Enable EXIF location extraction
            face_recognizer=face_recognizer  # Enable face recognition if available
        )
        timer.end("3. Total Image Processing")
        
        # Display results
        print("\n=== RESULTS ===")
        print(f"Model: {result.model_used}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"\nCaption:\n{result.caption}")
        print(f"\nTags ({len(result.tags)}):")
        for tag in result.tags:
            print(f"  - {tag}")
        
        # Print timing summary
        timer.print_summary()
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        
    except VisionModelError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        logger.exception("Unexpected error during processing")
        sys.exit(1)


if __name__ == "__main__":
    main()


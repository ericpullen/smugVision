#!/usr/bin/env python3
"""Debug face recognition to see why a face isn't being matched.

This script shows:
1. How many faces are detected
2. The best match for each face with distance scores
3. Whether the distance is within tolerance

Usage:
    python debug_face_recognition.py <image_path>
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Debug face recognition on an image."""
    if len(sys.argv) < 2:
        print("Usage: python debug_face_recognition.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    try:
        from smugvision.face import FaceRecognizer
        import face_recognition
        from PIL import Image, ImageOps
        import numpy as np
        
        # Register HEIF/HEIC support if available
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass
        
        # Load reference faces
        reference_faces_dir = Path.home() / ".smugvision" / "reference_faces"
        if not reference_faces_dir.exists():
            print(f"Error: Reference faces directory not found: {reference_faces_dir}")
            sys.exit(1)
        
        print(f"Loading reference faces from: {reference_faces_dir}")
        recognizer = FaceRecognizer(str(reference_faces_dir))
        print(f"✓ Loaded {len(recognizer.reference_faces)} person(s)")
        
        # Show what was loaded
        for person_name, encodings in recognizer.reference_faces.items():
            print(f"  - {person_name}: {len(encodings)} reference image(s)")
        
        print(f"\n{'='*70}")
        print(f"Analyzing image: {image_path}")
        print(f"{'='*70}\n")
        
        # Load image with EXIF orientation
        pil_image = Image.open(str(image_path))
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)
        
        # Detect faces using CNN model (more accurate for challenging lighting/angles)
        print("Detecting faces using CNN model (more accurate, slower)...")
        face_locations = face_recognition.face_locations(image, model="cnn")
        print(f"✓ Found {len(face_locations)} face(s)\n")
        
        if not face_locations:
            print("No faces detected in image.")
            return
        
        # Get face encodings using large model
        face_encodings = face_recognition.face_encodings(image, face_locations, model="large")
        
        # Analyze each face
        for i, (face_encoding, location) in enumerate(zip(face_encodings, face_locations), 1):
            top, right, bottom, left = location
            print(f"Face #{i} (location: top={top}, right={right}, bottom={bottom}, left={left})")
            print(f"{'-'*70}")
            
            # Compare with all reference faces
            matches = []
            
            for person_name, reference_encodings in recognizer.reference_faces.items():
                # Get minimum distance across all reference images for this person
                distances = []
                for ref_encoding in reference_encodings:
                    distance = face_recognition.face_distance([ref_encoding], face_encoding)[0]
                    distances.append(distance)
                
                min_distance = min(distances)
                matches.append((person_name, min_distance, len(distances)))
            
            # Sort by distance (best matches first)
            matches.sort(key=lambda x: x[1])
            
            # Show top 5 matches
            print(f"\nTop 5 matches for Face #{i}:")
            for j, (person_name, distance, ref_count) in enumerate(matches[:5], 1):
                within_tolerance = "✓ MATCH" if distance <= recognizer.tolerance else "✗ No match"
                confidence = (1.0 - (distance / recognizer.tolerance)) if distance <= recognizer.tolerance else 0.0
                print(f"  {j}. {person_name:20s} | distance: {distance:.4f} | {within_tolerance} | confidence: {confidence:.2%}")
                print(f"     (compared against {ref_count} reference image(s))")
            
            print(f"\nTolerance threshold: {recognizer.tolerance}")
            best_match = matches[0]
            if best_match[1] <= recognizer.tolerance:
                print(f"✓ Face #{i} identified as: {best_match[0]}")
            else:
                print(f"✗ Face #{i} not recognized (closest match: {best_match[0]} at distance {best_match[1]:.4f})")
                print(f"  To recognize this face, distance needs to be ≤ {recognizer.tolerance}")
                print(f"  Consider:")
                print(f"    - Adding more diverse reference images for {best_match[0]}")
                print(f"    - Increasing tolerance (currently {recognizer.tolerance})")
                print(f"    - Checking image quality and face angle")
            
            print()
        
        print(f"{'='*70}")
        print(f"\nSummary:")
        print(f"  Total faces detected: {len(face_locations)}")
        
        # Run the normal identify_faces to see what would be returned
        identified = recognizer.identify_faces(str(image_path))
        recognized_names = [name for name, conf in identified if name != "Unknown" and conf >= 0.35]
        print(f"  Faces identified: {len(recognized_names)}")
        if recognized_names:
            print(f"  Names: {', '.join(recognized_names)}")
        else:
            print(f"  Names: None")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nFace recognition not available. Install with:")
        print("  pip install face_recognition")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Unexpected error during face recognition debug")
        sys.exit(1)


if __name__ == "__main__":
    main()


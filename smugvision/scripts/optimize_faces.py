#!/usr/bin/env python3
"""Console script for optimizing reference face encodings.

This script resizes reference images to a reasonable size (800px max dimension)
and saves them as optimized JPEGs, which significantly speeds up face recognizer
initialization.

Usage:
    smugvision-optimize-faces [reference_faces_dir]
"""

import sys
from pathlib import Path
from PIL import Image, ImageOps

# Register HEIF/HEIC support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def optimize_reference_image(image_path: Path, max_dimension: int = 800) -> bool:
    """Optimize a reference face image.
    
    Args:
        image_path: Path to the image file
        max_dimension: Maximum width or height in pixels
        
    Returns:
        True if optimized, False if skipped or failed
    """
    try:
        # Skip if already a reasonably-sized JPEG
        if image_path.suffix.lower() == '.jpg' or image_path.suffix.lower() == '.jpeg':
            img = Image.open(image_path)
            if max(img.size) <= max_dimension:
                print(f"  ✓ Already optimized: {image_path.name}")
                return False
        
        # Load and process image
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if needed
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  ✓ Resized {image_path.name}: {img.size[0]}x{img.size[1]}")
        
        # Save as optimized JPEG (overwrites original)
        output_path = image_path.with_suffix('.jpg')
        img.save(output_path, 'JPEG', quality=90, optimize=True)
        
        # Remove original if it was a different format
        if output_path != image_path:
            image_path.unlink()
            print(f"  ✓ Converted {image_path.name} → {output_path.name}")
        else:
            print(f"  ✓ Optimized: {image_path.name}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to optimize {image_path.name}: {e}")
        return False


def main():
    """Optimize all reference face images."""
    if len(sys.argv) < 2:
        ref_dir = Path.home() / ".smugvision" / "reference_faces"
        print(f"No directory specified, using default: {ref_dir}")
    else:
        ref_dir = Path(sys.argv[1])
    
    if not ref_dir.exists():
        print(f"Error: Directory not found: {ref_dir}")
        print()
        print("To set up face recognition:")
        print("  1. Create the directory: mkdir -p ~/.smugvision/reference_faces")
        print("  2. Add person folders: mkdir ~/.smugvision/reference_faces/John_Doe")
        print("  3. Add face photos to each person's folder")
        print("  4. Run this script again to optimize them")
        return 1
    
    print(f"Optimizing reference faces in: {ref_dir}\n")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    total = 0
    optimized = 0
    
    # Process each person's directory
    for person_dir in sorted(ref_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        
        print(f"{person_dir.name}:")
        person_count = 0
        
        for image_path in sorted(person_dir.iterdir()):
            if not image_path.is_file():
                continue
            
            if image_path.suffix.lower() not in image_extensions:
                continue
            
            total += 1
            if optimize_reference_image(image_path):
                optimized += 1
                person_count += 1
        
        if person_count > 0:
            print(f"  ({person_count} image(s) optimized)\n")
        else:
            print()
    
    if total == 0:
        print("No reference face images found.")
        print()
        print("To add reference faces:")
        print("  1. Create person folders: mkdir ~/.smugvision/reference_faces/John_Doe")
        print("  2. Add clear face photos to each person's folder")
        print("  3. Run this script again to optimize them")
        return 0
    
    print(f"{'='*60}")
    print(f"Complete: {optimized}/{total} images optimized")
    print(f"\nExpected speedup: {(total * 0.12):.1f}s → {(total * 0.01):.1f}s")
    print(f"  (from ~0.12s to ~0.01s per image)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

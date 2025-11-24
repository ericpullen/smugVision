# Face Recognition Setup

smugVision supports face recognition to identify people in images. This feature uses the `face_recognition` library, which processes everything locally on your machine for privacy.

## Installation

1. Install the face recognition library:
```bash
pip install face_recognition
```

2. Install setuptools (required by models):
```bash
pip install setuptools
```

3. Install the required models:
```bash
pip install git+https://github.com/ageitgey/face_recognition_models
```

**Note:** On macOS, you may also need to install dlib dependencies:
```bash
brew install cmake
pip install dlib
```

**Troubleshooting:** 
- If you see an error about missing models, make sure you've installed:
  1. `face_recognition`
  2. `setuptools` 
  3. `face_recognition_models` (from GitHub)
- If you see "ModuleNotFoundError: No module named 'pkg_resources'", install setuptools: `pip install setuptools`

## Setting Up Reference Faces

1. Create a directory for reference face images:
```bash
mkdir -p ~/.smugvision/reference_faces
```

2. Add reference images of people you want to identify:
   - Name each image file with the person's name
   - Use underscores or hyphens for spaces (e.g., `John_Doe.jpg`, `Jane_Smith.png`)
   - You can have multiple images per person (e.g., `John_Doe_1.jpg`, `John_Doe_2.jpg`)
   - Supported formats: JPG, PNG, HEIC

Example directory structure:
```
~/.smugvision/reference_faces/
├── John_Doe_1.jpg
├── John_Doe_2.jpg
├── Jane_Smith.jpg
└── Bob_Johnson.png
```

## Usage

When you process images, smugVision will automatically:
1. Detect faces in the image
2. Compare them to your reference faces
3. Include identified person names in the caption prompt
4. Generate captions that naturally include the person's name

Example:
```python
from smugvision.face import FaceRecognizer
from smugvision.vision.factory import VisionModelFactory

# Load reference faces
face_recognizer = FaceRecognizer("~/.smugvision/reference_faces")

# Create vision model
model = VisionModelFactory.create("llama3.2-vision")

# Process image with face recognition
result = model.process_image(
    image_path="photo.jpg",
    caption_prompt="Describe this image...",
    tags_prompt="Generate tags...",
    face_recognizer=face_recognizer
)
```

## Tips for Best Results

1. **Reference Image Quality:**
   - Use clear, front-facing photos
   - Good lighting
   - Face should be clearly visible
   - Multiple angles/expressions help

2. **Tolerance Setting:**
   - Default tolerance is 0.6 (good balance)
   - Lower (0.4-0.5) = stricter, fewer false positives
   - Higher (0.7-0.8) = more lenient, may have false matches

3. **Privacy:**
   - All processing happens locally
   - No data is sent to external services
   - Reference faces are stored only on your machine

## Troubleshooting

**"No faces detected":**
- Ensure the reference image contains a clear face
- Try a different reference image

**"Unknown" faces:**
- Person may not be in your reference set
- Try adding more reference images of that person
- Check that reference images are clear and front-facing

**Installation issues:**
- On macOS, ensure Xcode command line tools are installed
- May need to install cmake: `brew install cmake`


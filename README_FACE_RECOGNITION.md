# Face Recognition Setup

smugVision can identify people in images and fold their names into the prompt sent to the
vision model, so generated captions contain real names. All processing happens locally.

## Choosing a backend

`face_recognition.backend` selects the embedding backend:

| Backend | Status | Install | Metric |
|---|---|---|---|
| `"dlib"` | **Default.** Well exercised. | Included in the core dependencies | Euclidean distance vs `tolerance` (lower is stricter) |
| `"insightface"` | **Optional and experimental.** | `pip install -e ".[insightface]"` | Cosine similarity vs `insightface.similarity_threshold` (higher is stricter) |

If the selected backend's dependencies are missing, smugVision logs an actionable error
and falls back to dlib rather than aborting the run. If no backend can be built at all,
face recognition is disabled with a warning and processing continues without names.

Whichever backend is active, `face_recognition.min_confidence` means the same thing: a
normalized 0.0-1.0 score where 0.0 sits exactly at that backend's own match threshold and
1.0 is a perfect match. `tolerance` and `similarity_threshold` are *not* interchangeable —
they are different metrics running in opposite directions.

Each backend keeps its own encoding cache file (the backend name and embedding dimension
are part of the filename and are verified in the cache manifest), so switching between
them does not corrupt or invalidate the other's cache.

### About the InsightFace backend

It is marked experimental deliberately. What has been verified: it builds, encodes
reference faces, produces correctly-normalized 512-d embeddings, respects the shared
confidence scale, and agreed with dlib on every recognized name in a small sample. It also
detected *more* faces than dlib on that sample (SCRFD picks up small background faces), so
detected-face counts will legitimately change if you switch.

What has **not** been verified: any accuracy claim. Its reputed advantage on profile
angles, unusual lighting and age variation was not benchmarked here. Treat a switch as an
experiment to evaluate on your own library, not an upgrade.

Two practical notes:

- The model pack named by `insightface.model_name` (default `buffalo_l`) downloads on
  first use — a few hundred MB into `~/.insightface/models`.
- Reference images with no detectable face are retried on every run, which forces full
  ONNX session initialization even with a warm cache. On one real reference set (309
  images, 7 of which have no detectable face) that cost 4.3-4.6s per recognizer
  construction versus 0.01s for dlib.

## Installation (dlib backend)

The `face-recognition` package is a core dependency, so `pip install -e .` normally covers
it. If face recognition reports itself unavailable:

1. Install the face recognition library:
```bash
pip install face_recognition
```

2. Install setuptools (required by the models package):
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
- If you see `ModuleNotFoundError: No module named 'pkg_resources'`, install setuptools:
  `pip install setuptools`. Note that **setuptools 81 removed `pkg_resources`**, which
  `face_recognition_models` imports — so on a modern environment the fix is
  `pip install "setuptools<81"`, not simply installing the latest.
- `face_recognition` calls `quit()` rather than raising `ImportError` when its models are
  missing. smugVision catches that too, so a broken install shows up as "face recognition
  disabled" in the log rather than a silent exit.

## Installation (InsightFace backend)

```bash
pip install -e ".[insightface]"
```

Then set `face_recognition.backend: "insightface"` in `~/.smugvision/config.yaml`.

## Setting Up Reference Faces

Reference faces are organized as **one subdirectory per person**. The directory name is
the person's name; underscores are converted to spaces for display and in captions.

```bash
mkdir -p ~/.smugvision/reference_faces
```

```
~/.smugvision/reference_faces/
├── John_Doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── vacation.png
├── Jane_Smith/
│   ├── profile.jpg
│   └── headshot.heic
└── Bob_Johnson/
    └── portrait.jpg
```

- Supported formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`
- Multiple clear photos per person improve accuracy
- `smugvision-optimize-faces` downscales reference images in place, which speeds up
  encoding considerably. Note it changes their size and mtime, so it invalidates the
  encoding cache and forces one re-encode.

The underscore form is significant: `~/.smugvision/relationships.yaml` is matched against
the raw directory names (`John_Doe`), and the vision layer converts to "John Doe" only
when building the prompt text.

## Usage

When you process images, smugVision automatically:
1. Detects faces in the image
2. Compares them to your reference faces
3. Includes identified person names — and the total number of faces detected — in the
   prompt, so the model can say "one of them is X" when it sees more faces than it can name
4. Generates captions that naturally include the person's name

```python
from pathlib import Path

from smugvision.config import ConfigManager
from smugvision.face import FaceRecognizer
from smugvision.vision import VisionModelFactory

config = ConfigManager.load()

# Options come from the sub-block named after the selected backend, exactly as
# ImageProcessor does it - handing the dlib backend an `insightface` block would
# only log "Ignoring unsupported dlib backend option: ...".
backend = config.get("face_recognition.backend", "dlib") or "dlib"

face_recognizer = FaceRecognizer(
    str(Path("~/.smugvision/reference_faces").expanduser()),
    backend=backend,
    backend_options=config.get(f"face_recognition.{backend}", {}) or {},
)

model = VisionModelFactory.create(config.get("vision.model"))

image_path = "photo.jpg"
raw_names = face_recognizer.get_person_names(image_path, min_confidence=0.25)

result = model.generate_metadata(
    image_path,
    config.get("prompts.caption"),
    config.get("prompts.tags"),
    person_names=raw_names,                       # raw, underscores intact
    total_faces=face_recognizer.get_face_count(image_path),
    location_context="Louisville, Kentucky",
    album_name="Summer 2026",
)

print(result.caption, result.tags)
```

Pass `person_names` with underscores intact — the vision layer formats them for the model
and `relationships.yaml` is matched on the underscore form. `total_faces` is what enables
the "there are N people, one of them is X" phrasing; omit it and the model only ever hears
about the people it can name.

`VisionModel.process_image(image_path, caption_prompt, tags_prompt, ...,
face_recognizer=...)` still exists as a convenience wrapper that does the face and EXIF
lookups for you.

## Tips for Best Results

1. **Reference Image Quality:**
   - Use clear, front-facing photos
   - Good lighting
   - Face should be clearly visible
   - Multiple angles/expressions help

2. **Threshold settings:**
   - dlib `tolerance`: default 0.6. Lower (0.4-0.5) = stricter, fewer false positives;
     higher (0.7-0.8) = more lenient, more false matches.
   - InsightFace `insightface.similarity_threshold`: default 0.4. **Higher** is stricter.
   - `min_confidence` (default 0.25) applies on top of either, on the shared normalized
     scale.
   - Changing a threshold does not force a re-encode — thresholds affect how vectors are
     compared, not how they are computed.

3. **Privacy:**
   - All processing happens locally
   - No data is sent to external services
   - Reference faces are stored only on your machine

## Troubleshooting

**"No faces detected":**
- Ensure the reference image contains a clear face
- Try a different reference image
- Reference images with no detectable face are skipped with a warning on every run

**"Unknown" faces:**
- Person may not be in your reference set
- Try adding more reference images of that person
- Check that reference images are clear and front-facing

**Installation issues:**
- On macOS, ensure Xcode command line tools are installed
- May need to install cmake: `brew install cmake`
- See the setuptools 81 note above

**`tests/debug_face_recognition.py` raises a shape error:**
- That script is dlib-only. It calls `face_recognition.face_distance` directly on the
  stored vectors, which are 512-d under the InsightFace backend rather than 128-d.

# SmugVision Performance Analysis

This file holds two things:

1. **[Current numbers](#current-numbers-2026-08-11)** — measurements taken after the
   single-structured-call rewrite of the vision layer.
2. **[Original investigation](#original-investigation-historical)** — the earlier
   analysis that identified reverse geocoding as the dominant cost. Its recommendations
   have since shipped; it is kept for context, not as a to-do list.

---

## Current numbers (2026-08-11)

### How these were measured — read before quoting them

- Model: **`gemma4:latest`** (the e4b variant) on Ollama 0.32.9, local, Apple Silicon.
- Images: real cached originals from `~/.smugvision/cache/`, e.g. a 3840x2880 / 8.12 MB JPEG.
- These are **vision-inference timings only**. They do not include download, geocoding,
  or face recognition.
- The historical 18s + 6s figures below were taken on a **different model** on a
  **different machine state**. The old and new numbers are therefore *not* a controlled
  A/B of one-call vs two-call — the honest claim is "one structured call on this model
  costs ~2-3s", not "the rewrite made inference 8x faster".

### Vision inference per image

| Configuration | Measured (warm) | Notes |
|---|---|---|
| `single_call: true`, `structured_output: true` (default) | **1.76 – 2.77s** | One image encode, one `chat` call, JSON-Schema constrained |
| `single_call: true`, `structured_output: false` | 2.26s warm; **22.7s** on one cold run | Unconstrained free text rambles toward `num_predict` |
| `single_call: false`, `structured_output: true` | 3.24 – 3.31s | Two requests, two encodes |
| `single_call: false`, `structured_output: false` | 3.25 – 4.00s | Fully legacy path |
| Legacy processor call shape (`generate_caption` then `generate_tags`, `max_tokens=500`) | 3.74s | Same model, for comparison against the row above |

A direct probe of the raw `chat` call measured **9.57s on the first (cold) request** for a
new image, then **1.21 – 1.26s** on repeats (`eval_count` 50-61, `done_reason=stop`).
That cold/warm gap is what `vision.keep_alive` addresses: without it Ollama can unload
the model between images, so every image pays the cold cost rather than just the first.

Independent single-call measurement by a second observer on the same setup: **2.9s** for
caption + tags in one structured request.

### `think` (reasoning) settings

| `vision.think` | Result |
|---|---|
| `false` (default) | 1.80s |
| `"low"` | 9.14s, and **only with `max_tokens: 1200`** — at 400 tokens the model spent the entire budget reasoning and returned no content |
| `null` (omitted) | 4.64s — gemma4 reasons by default |

This is why `vision.think: false` is the shipped default, and why "raise `max_tokens`" is
the wrong first move when replies come back empty.

### Image downscaling (`vision.max_image_dimension`)

Same 3840x2880 source JPEG, base64 payload size:

| Setting | Payload |
|---|---|
| disabled (`0` / `null`) | 5.59 MB |
| `1568` (default) | 1.07 MB (**5.2x smaller**) |
| `800` | 0.27 MB |

Vision models tile input to roughly 1024-1568px, so the extra pixels were being paid for
and then discarded. `max_dimension` never upscales.

### Face recognition (dlib backend, 309 reference images across 12 people)

| Operation | Measured |
|---|---|
| Reference encoding, cold (cache rebuild) | 21.7s |
| Reference load from warm cache | 0.01s |
| Repeat detection on the same image (memoized) | 0.43s → ~0.000s |

The memo matters because `get_person_names()` and `get_face_count()` used to run full
detection independently on the same image.

**InsightFace backend (optional, experimental):** constructing the recognizer with a warm
cache measured **4.3 – 4.6s** versus 0.01s for dlib, because 7 reference images have no
detectable face, are not recorded in the manifest, and are retried every run — which forces
full ONNX session initialization. Accuracy was *not* benchmarked; see
`README_FACE_RECOGNITION.md`.

### What has not been measured

- End-to-end `smugvision --gallery <key> --dry-run` wall-clock after these changes.
- Any model other than `gemma4:latest`.
- InsightFace vs dlib recognition accuracy.

Do not extrapolate a total-per-image figure from the table above; the download and
geocoding phases dominate and have not been re-measured since the original investigation.

---

## Original investigation (historical)

> Kept for context. The reverse-geocoding rewrite described under "Recommendations"
> has since shipped — `resolve_location_with_custom()` now checks
> `~/.smugvision/locations.yaml` first and falls back to Nominatim plus a single
> Overpass POI lookup. The line references below point at code that has changed and
> have not been re-verified.

Based on the log output analysis, here's where time was being spent during image processing:

### Timing Breakdown (from logs)

| Phase | Duration | Percentage | Issue |
|-------|----------|------------|-------|
| **GPS Reverse Geocoding #1** | ~47s | 35% | ⚠️ MAJOR BOTTLENECK (since fixed) |
| User venue selection (interactive) | ~28s | 21% | Expected (user input) |
| Face detection | ~6s | 4.5% | Reasonable |
| **GPS Reverse Geocoding #2** | ~47s | 35% | ⚠️ MAJOR BOTTLENECK (since fixed) |
| Caption generation | ~18s | 13% | Superseded — see [Current numbers](#current-numbers-2026-08-11) |
| Tags generation | ~6s | 4.5% | Superseded — caption and tags are now one call |
| Face recognizer init | ~1s | <1% | Reasonable |
| Model initialization | <1s | <1% | Reasonable |

**Total Processing Time: ~153s (2.5 minutes)**
**Actual Processing (excluding user input): ~125s**

### 🔴 Critical Issue: Reverse Geocoding Taking 47 Seconds — RESOLVED

**Location:** `smugvision/utils/exif.py` (line numbers from the original analysis no
longer apply)

**Problem:** the `reverse_geocode()` function iterated through ~40 different venue types,
making a separate Nominatim call for each with a 5-second timeout. Half the list being
tried meant 20+ calls × 5s = 100s+.

**Resolution:** replaced by a single reverse geocode plus one Overpass POI query, with
`~/.smugvision/locations.yaml` short-circuiting known places entirely
(`location.check_custom_first`). The post-fix wall-clock has not been re-measured.

### Original recommendations

1. Use a single Overpass/Nominatim query instead of per-venue-type searches — **shipped**
2. Cache results so the same coordinates are not resolved twice — **not done**. The only
   `_geocode_cache` in the tree is in `smugvision/utils/exif_optimized.py`, which nothing
   imports. The live path, `utils/exif.py::reverse_geocode()`, builds a fresh `Nominatim`
   geolocator and reverse-geocodes on **every** call, so an album of N photos shot at one
   venue still costs N reverse lookups plus N Overpass queries.
3. Reduce the 5s per-venue timeout — **moot**, the loop is gone
4. Limit the venue type list — **moot**, the loop is gone
5. Parallelize with `ThreadPoolExecutor` — **not done**, no longer needed

The projected "153s → ~35s" figure from the original analysis was an estimate, never a
measurement. It should not be quoted as a result.

## Monitoring

`tests/test_vision.py` prints a timing breakdown per phase:

```bash
python tests/test_vision.py <image_path>
```

Note that this script hardcodes a model name; edit it to match your `vision.model` (or
any name from `ollama list`) before drawing conclusions from it.

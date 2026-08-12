#!/usr/bin/env python3
"""Benchmark Ollama vision models through the real smugVision vision layer.

The harness answers one question: which locally installed model should be the
``vision.model`` default? It samples photos from the user's own cached albums, gathers
the same context the processor would supply (album name, recognized people, resolved
location) and then calls :meth:`VisionModel.generate_metadata` - the real refactored code
path, not a hand-rolled Ollama call - once per (model, image) pair.

Timing notes:

* The first call for each model is treated as warm-up and excluded from the statistics,
  because it pays for loading the weights into memory.
* Face recognition and location resolution run ONCE per image, before any model is
  benchmarked, so their cost never lands in a model's timing.

The script is strictly read-only with respect to user state: it never writes to SmugMug
and never modifies ``~/.smugvision/config.yaml``. Models that are not installed are
skipped with the exact ``ollama pull`` command; nothing is ever pulled automatically.

Examples:
    Benchmark every installed vision-capable model over four cached photos::

        python scripts/benchmark_models.py

    Compare two specific models over eight photos and keep the raw results::

        python scripts/benchmark_models.py --models gemma4:latest llava:7b \\
            --samples 8 --output /tmp/vision-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_COUNT = 4
DEFAULT_OUTPUT = Path("benchmark_results.json")

# Image extensions worth feeding to a vision model.
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}
)

# Sub-directories of the cache that hold state rather than photos.
NON_ALBUM_DIRS = frozenset({"face_encodings"})

# Warning fragments emitted by smugvision.vision.llama when a structured response could
# not be parsed cleanly on the first attempt.
_SALVAGE_ATTEMPTED = "was unusable"
_SALVAGE_SUCCEEDED = "Salvaged partial JSON"
_FREETEXT_FALLBACK = "Falling back to free-text parsing"


@dataclass
class SampleImage:
    """A photo selected for benchmarking, with its pre-computed context.

    Attributes:
        path: Absolute path to the image file
        album: Album (directory) name the photo came from
        person_names: Recognized reference-face names, raw ``John_Doe`` form
        total_faces: Number of faces detected, which may exceed len(person_names)
        location_context: Resolved place name, or None when unknown
    """

    path: Path
    album: str
    person_names: List[str] = field(default_factory=list)
    total_faces: int = 0
    location_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Render the sample as JSON-serializable data.

        Returns:
            Dictionary with the path stringified
        """
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass
class CallResult:
    """The outcome of a single generate_metadata call.

    Attributes:
        model: Model name that produced the result
        image: Absolute path to the image
        album: Album the image came from
        warm_up: True when this call is excluded from the timing statistics
        elapsed: Wall-clock seconds for the call, or None when it failed
        caption: Generated caption text
        tags: Generated keyword tags
        parse_mode: How the response was interpreted - "structured" (schema-constrained
            JSON parsed on the first attempt), "salvaged", "freetext",
            "freetext-configured" or "error"
        warnings: Warning messages emitted by the vision layer during the call
        error: Exception text when the call failed, otherwise None
    """

    model: str
    image: str
    album: str
    warm_up: bool
    elapsed: Optional[float]
    caption: str = ""
    tags: List[str] = field(default_factory=list)
    parse_mode: str = "structured"
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """Whether the call returned usable metadata.

        Returns:
            True if no exception was raised
        """
        return self.error is None


@dataclass
class ModelSummary:
    """Aggregated timing and reliability statistics for one model.

    Attributes:
        model: Model name
        calls: Total calls attempted
        succeeded: Number of calls that returned metadata
        failed: Number of calls that raised
        warm_up_seconds: Elapsed time of the discarded warm-up call, if any
        timed_calls: Number of calls included in the statistics
        median: Median seconds across timed calls
        mean: Mean seconds across timed calls
        fastest: Fastest timed call in seconds
        slowest: Slowest timed call in seconds
        clean_json: Timed-or-not calls whose structured JSON parsed on the first attempt
        avg_caption_chars: Mean caption length in characters
        avg_tags: Mean number of tags returned
    """

    model: str
    calls: int
    succeeded: int
    failed: int
    warm_up_seconds: Optional[float]
    timed_calls: int
    median: Optional[float]
    mean: Optional[float]
    fastest: Optional[float]
    slowest: Optional[float]
    clean_json: int
    avg_caption_chars: Optional[float]
    avg_tags: Optional[float]


class _ParseModeWatcher(logging.Handler):
    """Capture vision-layer warnings so parse fallbacks can be reported.

    The vision layer already logs whenever a structured response has to be salvaged or
    handed to the free-text parser. Listening to those records is cheaper - and far less
    invasive - than threading a status flag through the public API.
    """

    def __init__(self) -> None:
        """Initialize the handler at WARNING level."""
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Record a formatted warning message.

        Args:
            record: Log record emitted by the vision layer
        """
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - a logging handler must never raise
            pass

    def reset(self) -> None:
        """Drop everything captured so far."""
        self.messages.clear()

    def parse_mode(self, structured: bool) -> str:
        """Classify how the most recent response was interpreted.

        Args:
            structured: Whether structured output (a JSON schema) was requested

        Returns:
            One of "structured", "salvaged", "freetext" or "freetext-configured"
        """
        if not structured:
            return "freetext-configured"
        joined = "\n".join(self.messages)
        if _FREETEXT_FALLBACK in joined:
            return "freetext"
        if _SALVAGE_SUCCEEDED in joined or _SALVAGE_ATTEMPTED in joined:
            return "salvaged"
        return "structured"


def load_config(config_path: Optional[str]) -> Any:
    """Load the smugVision configuration without ever writing it back.

    Args:
        config_path: Explicit path to a config file, or None for the default search order

    Returns:
        A ConfigManager instance

    Raises:
        SystemExit: If the configuration cannot be loaded
    """
    from smugvision.config.manager import ConfigManager

    try:
        return ConfigManager.load(
            config_path=config_path, interactive=False, create_if_missing=False
        )
    except Exception as e:
        print(f"ERROR: could not load configuration: {e}", file=sys.stderr)
        raise SystemExit(2) from e


def discover_installed_models(endpoint: Optional[str], timeout: float = 10.0) -> List[str]:
    """List vision-capable models the local Ollama server can serve right now.

    Args:
        endpoint: Ollama endpoint URL, or None for the client default
        timeout: Seconds to wait for the server

    Returns:
        Sorted list of installed vision-capable model names, possibly empty
    """
    try:
        import ollama

        from smugvision.vision.factory import VisionModelFactory
        from smugvision.vision.llama import LlamaVisionModel

        client = ollama.Client(host=endpoint, timeout=timeout)
        names = LlamaVisionModel._extract_model_names(client.list())
        vision = [name for name in names if VisionModelFactory._is_vision_capable(client, name)]
        return sorted(vision)
    except Exception as e:
        logger.warning(f"Could not list models from Ollama: {e}")
        return []


def album_directories(cache_dir: Path, wanted: Optional[Sequence[str]]) -> List[Path]:
    """Find album directories under the cache directory.

    Args:
        cache_dir: Root cache directory (``~/.smugvision/cache`` by default)
        wanted: Optional album names or case-insensitive substrings to restrict to

    Returns:
        Sorted list of album directories that contain at least one image
    """
    if not cache_dir.is_dir():
        return []

    albums: List[Path] = []
    for entry in sorted(cache_dir.iterdir()):
        if not entry.is_dir() or entry.name in NON_ALBUM_DIRS or entry.name.startswith("."):
            continue
        if not any(p.suffix.lower() in IMAGE_SUFFIXES for p in entry.iterdir()):
            continue
        if wanted and not any(w.lower() in entry.name.lower() for w in wanted):
            continue
        albums.append(entry)
    return albums


def images_in(directory: Path) -> List[Path]:
    """List image files directly inside a directory.

    Args:
        directory: Directory to scan (not recursive)

    Returns:
        Sorted list of image paths
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def select_samples(
    cache_dir: Path,
    count: int,
    albums: Optional[Sequence[str]] = None,
    image_dir: Optional[Path] = None,
) -> Tuple[List[Path], List[Path]]:
    """Choose sample photos, spread across as many different albums as possible.

    Photos are taken round-robin - one per album, then a second per album, and so on -
    so a four-image run covers four scene types rather than four frames of one event.
    Selection is deterministic so repeated runs compare the same photos.

    Args:
        cache_dir: Root of the image cache
        count: Number of samples wanted
        albums: Optional album names/substrings to restrict the search to
        image_dir: Optional single directory to sample from instead of the cache

    Returns:
        Tuple of (selected paths, remaining unselected paths in round-robin order)
    """
    if image_dir is not None:
        pool = images_in(image_dir)
        return pool[:count], pool[count:]

    per_album = [images_in(album) for album in album_directories(cache_dir, albums)]
    ordered: List[Path] = []
    depth = 0
    while any(len(files) > depth for files in per_album):
        for files in per_album:
            if len(files) > depth:
                ordered.append(files[depth])
        depth += 1

    return ordered[:count], ordered[count:]


def build_face_recognizer(config: Any, backend_override: Optional[str]) -> Optional[Any]:
    """Construct a FaceRecognizer the same way ImageProcessor does.

    Args:
        config: Loaded ConfigManager
        backend_override: Backend to use instead of the configured one, or None.
            This is an in-memory override; the config file is never modified.

    Returns:
        A FaceRecognizer, or None when face recognition is disabled or unavailable
    """
    if not config.get("face_recognition.enabled", True):
        logger.info("Face recognition disabled in config; benchmarking without person context")
        return None

    reference_dir = Path(
        config.get("face_recognition.reference_faces_dir", "~/.smugvision/reference_faces")
    ).expanduser()
    if not reference_dir.exists():
        logger.warning(f"Reference faces directory not found at {reference_dir}")
        return None

    backend = backend_override or config.get("face_recognition.backend", "dlib") or "dlib"
    backend_options = config.get(f"face_recognition.{backend}", {}) or {}
    cache_dir = Path(
        config.get("face_recognition.cache_dir", "~/.smugvision/cache/face_encodings")
    ).expanduser()

    try:
        from smugvision.face.recognizer import FaceRecognizer

        recognizer = FaceRecognizer(
            str(reference_dir),
            cache_dir=str(cache_dir),
            use_cache=config.get("face_recognition.use_cache", True),
            backend=backend,
            backend_options=backend_options,
        )
        logger.info(
            f"Face recognition enabled ({backend}) with "
            f"{len(recognizer.reference_faces)} reference person(s)"
        )
        return recognizer
    except Exception as e:
        logger.warning(f"Could not initialize face recognizer ({backend}): {e}")
        return None


def resolve_location(
    image_path: Path, album: str, album_locations: Dict[str, str]
) -> Optional[str]:
    """Determine the location context for a photo.

    ``--album-location`` overrides win; otherwise EXIF GPS is read and reverse geocoded.
    Note that SmugMug strips GPS from downloaded image bytes, so cached photos usually
    carry no coordinates and an override is the only way to exercise the place-name path.

    Args:
        image_path: Path to the image
        album: Album name the image belongs to
        album_locations: Mapping of album substring (lowercased) to place name

    Returns:
        A place name, or None when no location could be determined
    """
    for fragment, place in album_locations.items():
        if fragment in album.lower():
            return place

    try:
        from smugvision.utils.exif import extract_exif_location, reverse_geocode

        location = extract_exif_location(str(image_path))
        if not location.has_coordinates or location.latitude is None or location.longitude is None:
            return None
        name = reverse_geocode(location.latitude, location.longitude, interactive=False)
        return name or f"{location.latitude:.6f}, {location.longitude:.6f}"
    except Exception as e:
        logger.debug(f"Location lookup failed for {image_path.name}: {e}")
        return None


def gather_context(
    paths: Sequence[Path],
    recognizer: Optional[Any],
    album_locations: Dict[str, str],
    min_confidence: float,
) -> List[SampleImage]:
    """Compute per-image context once, ahead of any model call.

    Args:
        paths: Images to describe
        recognizer: Optional FaceRecognizer for person names
        album_locations: Mapping of album substring (lowercased) to place name
        min_confidence: Minimum face-match confidence for a name to be used

    Returns:
        List of SampleImage in the same order as ``paths``
    """
    samples: List[SampleImage] = []
    for path in paths:
        album = path.parent.name
        names: List[str] = []
        total_faces = 0
        if recognizer is not None:
            try:
                names = recognizer.get_person_names(str(path), min_confidence=min_confidence)
                total_faces = recognizer.get_face_count(str(path))
            except Exception as e:
                logger.warning(f"Face recognition failed for {path.name}: {e}")
        samples.append(
            SampleImage(
                path=path,
                album=album,
                person_names=names,
                total_faces=total_faces,
                location_context=resolve_location(path, album, album_locations),
            )
        )
    return samples


def ensure_people_sample(
    samples: List[SampleImage],
    remaining: Sequence[Path],
    recognizer: Optional[Any],
    album_locations: Dict[str, str],
    min_confidence: float,
) -> List[SampleImage]:
    """Guarantee that at least one sample contains recognized people.

    A model's handling of supplied person names is one of the things being judged, so a
    sample set of landscapes would not exercise it. If no selected photo has a recognized
    face, unselected candidates are scanned until one does and it replaces the last sample.

    Args:
        samples: Samples chosen so far, each already carrying its context
        remaining: Unselected candidate paths, in preference order
        recognizer: Optional FaceRecognizer; without one this is a no-op
        album_locations: Mapping of album substring (lowercased) to place name
        min_confidence: Minimum face-match confidence for a name to be used

    Returns:
        The (possibly amended) sample list
    """
    if recognizer is None or not samples or any(s.person_names for s in samples):
        return samples

    logger.info("No recognized people in the initial sample; scanning for a photo with faces")
    for path in remaining:
        try:
            names = recognizer.get_person_names(str(path), min_confidence=min_confidence)
        except Exception as e:
            logger.debug(f"Face recognition failed for {path.name}: {e}")
            continue
        if names:
            replacement = gather_context([path], recognizer, album_locations, min_confidence)[0]
            logger.info(
                f"Substituting {path.name} ({', '.join(names)}) to cover the person-name path"
            )
            samples[-1] = replacement
            break
    else:
        logger.warning("No cached photo produced a recognized person; person context untested")
    return samples


def create_model(model_name: str, config: Any, args: argparse.Namespace) -> Any:
    """Instantiate the real smugVision vision model for a model name.

    Every option is taken from the user's config so the benchmark measures the code path
    the CLI would actually run, with only the model name substituted.

    Args:
        model_name: Ollama model name
        config: Loaded ConfigManager
        args: Parsed command-line arguments

    Returns:
        A VisionModel instance

    Raises:
        VisionModelError: If the model cannot be constructed
    """
    from smugvision.vision.factory import VisionModelFactory

    return VisionModelFactory.create(
        model_name,
        endpoint=args.endpoint or config.get("vision.endpoint"),
        timeout=args.timeout or config.get("vision.timeout", 120),
        think=config.get("vision.think", False),
        keep_alive=config.get("vision.keep_alive", "30m"),
        single_call=config.get("vision.single_call", True),
        structured_output=config.get("vision.structured_output", True),
        max_image_dimension=config.get("vision.max_image_dimension", 1568),
        jpeg_quality=config.get("vision.jpeg_quality", 85),
        validate_model=False,
    )


def unload_model(model: Any) -> None:
    """Ask Ollama to evict a model from memory, best effort.

    Large models are benchmarked back to back; releasing one before loading the next keeps
    the machine from swapping and makes the next model's warm-up representative.

    Args:
        model: The VisionModel whose weights should be released
    """
    try:
        model.client.chat(model=model.model_name, messages=[], keep_alive=0)
        logger.debug(f"Requested unload of {model.model_name}")
    except Exception as e:
        logger.debug(f"Could not unload {model.model_name}: {e}")


def benchmark_model(
    model_name: str,
    samples: Sequence[SampleImage],
    config: Any,
    args: argparse.Namespace,
    watcher: _ParseModeWatcher,
) -> List[CallResult]:
    """Run every sample through one model, timing each call.

    Args:
        model_name: Ollama model name to benchmark
        samples: Images with their pre-computed context
        config: Loaded ConfigManager, source of the prompts and vision options
        args: Parsed command-line arguments
        watcher: Handler capturing vision-layer warnings

    Returns:
        One CallResult per sample, in order. The first is flagged as warm-up.
    """
    caption_prompt = config.get("prompts.caption", "")
    tags_prompt = config.get("prompts.tags", "")
    temperature = (
        args.temperature if args.temperature is not None else config.get("vision.temperature", 0.4)
    )
    max_tokens = (
        args.max_tokens if args.max_tokens is not None else config.get("vision.max_tokens", 400)
    )
    structured = bool(config.get("vision.structured_output", True))

    print(f"\n>>> {model_name}", flush=True)
    try:
        model = create_model(model_name, config, args)
    except Exception as e:
        print(f"    FAILED to initialize: {e}", flush=True)
        return [
            CallResult(
                model=model_name,
                image=str(s.path),
                album=s.album,
                warm_up=index == 0,
                elapsed=None,
                parse_mode="error",
                error=f"model initialization failed: {e}",
            )
            for index, s in enumerate(samples)
        ]

    results: List[CallResult] = []
    for index, sample in enumerate(samples):
        warm_up = index == 0
        label = f"    [{index + 1}/{len(samples)}] {sample.path.name}"
        watcher.reset()
        start = time.perf_counter()
        try:
            metadata = model.generate_metadata(
                str(sample.path),
                caption_prompt,
                tags_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                location_context=sample.location_context,
                person_names=sample.person_names or None,
                total_faces=sample.total_faces or None,
                album_name=sample.album,
            )
            elapsed = time.perf_counter() - start
            results.append(
                CallResult(
                    model=model_name,
                    image=str(sample.path),
                    album=sample.album,
                    warm_up=warm_up,
                    elapsed=elapsed,
                    caption=metadata.caption,
                    tags=list(metadata.tags),
                    parse_mode=watcher.parse_mode(structured),
                    warnings=list(watcher.messages),
                )
            )
            suffix = " (warm-up, excluded)" if warm_up else ""
            print(f"{label}: {elapsed:.2f}s, {len(metadata.tags)} tags{suffix}", flush=True)
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append(
                CallResult(
                    model=model_name,
                    image=str(sample.path),
                    album=sample.album,
                    warm_up=warm_up,
                    elapsed=None,
                    parse_mode="error",
                    warnings=list(watcher.messages),
                    error=str(e),
                )
            )
            print(f"{label}: FAILED after {elapsed:.2f}s - {e}", flush=True)

    if args.unload:
        unload_model(model)
    return results


def summarize(results: Sequence[CallResult], discard_warm_up: bool) -> ModelSummary:
    """Aggregate one model's calls into timing and quality statistics.

    Args:
        results: All calls for a single model
        discard_warm_up: Whether the first call is excluded from the timing statistics

    Returns:
        A ModelSummary. Timing fields are None when nothing timed successfully.
    """
    model = results[0].model if results else "unknown"
    successes = [r for r in results if r.succeeded]
    warm_up = next((r for r in results if r.warm_up), None)
    timed = [r for r in successes if not (discard_warm_up and r.warm_up)]
    durations = [r.elapsed for r in timed if r.elapsed is not None]

    return ModelSummary(
        model=model,
        calls=len(results),
        succeeded=len(successes),
        failed=len(results) - len(successes),
        warm_up_seconds=warm_up.elapsed if warm_up is not None else None,
        timed_calls=len(durations),
        median=statistics.median(durations) if durations else None,
        mean=statistics.fmean(durations) if durations else None,
        fastest=min(durations) if durations else None,
        slowest=max(durations) if durations else None,
        clean_json=sum(1 for r in successes if r.parse_mode == "structured"),
        avg_caption_chars=(
            statistics.fmean([len(r.caption) for r in successes]) if successes else None
        ),
        avg_tags=statistics.fmean([len(r.tags) for r in successes]) if successes else None,
    )


def _fmt(value: Optional[float], places: int = 2) -> str:
    """Format an optional float for the comparison table.

    Args:
        value: Value to format, possibly None
        places: Decimal places

    Returns:
        The formatted number, or "-" when the value is None
    """
    return "-" if value is None else f"{value:.{places}f}"


def print_table(summaries: Sequence[ModelSummary], discard_warm_up: bool) -> None:
    """Print the model comparison table.

    Args:
        summaries: One summary per benchmarked model
        discard_warm_up: Whether warm-up calls were excluded from the statistics
    """
    header = (
        f"{'MODEL':<20} {'OK/RUN':>7} {'WARMUP':>8} {'MEDIAN':>8} {'MEAN':>8} "
        f"{'MIN':>7} {'MAX':>7} {'JSON':>7} {'CAPCH':>7} {'TAGS':>6}"
    )
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda x: (x.median is None, x.median or 0.0)):
        print(
            f"{s.model:<20} {f'{s.succeeded}/{s.calls}':>7} {_fmt(s.warm_up_seconds):>8} "
            f"{_fmt(s.median):>8} {_fmt(s.mean):>8} {_fmt(s.fastest):>7} {_fmt(s.slowest):>7} "
            f"{f'{s.clean_json}/{s.succeeded}':>7} {_fmt(s.avg_caption_chars, 0):>7} "
            f"{_fmt(s.avg_tags, 1):>6}"
        )
    print("-" * len(header))
    print("Seconds per image, wall clock, through VisionModel.generate_metadata().")
    if discard_warm_up:
        print(
            "WARMUP is the first call per model (model load) and is EXCLUDED from "
            "MEDIAN/MEAN/MIN/MAX."
        )
    else:
        print("Warm-up discarding is disabled: every call is included in the statistics.")
    print("JSON = responses whose schema-constrained JSON parsed on the first attempt.")
    print("CAPCH = mean caption length in characters. TAGS = mean tag count.")


def print_samples(samples: Sequence[SampleImage]) -> None:
    """Print the sample set and the context supplied for each photo.

    Args:
        samples: Images selected for the run
    """
    print("\n" + "=" * 78)
    print("SAMPLE IMAGES AND CONTEXT SUPPLIED TO EVERY MODEL")
    print("=" * 78)
    for index, sample in enumerate(samples, start=1):
        people = ", ".join(n.replace("_", " ") for n in sample.person_names) or "(none recognized)"
        print(f"{index}. {sample.path.name}   [album: {sample.album}]")
        print(f"   faces detected: {sample.total_faces}   people: {people}")
        print(f"   location: {sample.location_context or '(none)'}")


def print_outputs(results: Sequence[CallResult], samples: Sequence[SampleImage]) -> None:
    """Print every caption and tag list so quality can be judged by eye.

    Results are grouped by image rather than by model, which is what makes side-by-side
    comparison of the same scene possible.

    Args:
        results: All calls from the run
        samples: Images used, in display order
    """
    print("\n" + "=" * 78)
    print("GENERATED OUTPUT, GROUPED BY IMAGE")
    print("=" * 78)
    for index, sample in enumerate(samples, start=1):
        people = ", ".join(n.replace("_", " ") for n in sample.person_names) or "none"
        print(f"\n--- Image {index}: {sample.path.name} [{sample.album}]")
        print(f"    context -> people: {people} | location: {sample.location_context or 'none'}")
        for result in results:
            if result.image != str(sample.path):
                continue
            flag = " (warm-up)" if result.warm_up else ""
            if not result.succeeded:
                print(f"\n  {result.model}{flag}: FAILED - {result.error}")
                continue
            timing = _fmt(result.elapsed)
            print(f"\n  {result.model}{flag}  [{timing}s, parse={result.parse_mode}]")
            print(f"    caption: {result.caption}")
            print(f"    tags:    {', '.join(result.tags) if result.tags else '(none)'}")


def write_results(
    output_path: Path,
    samples: Sequence[SampleImage],
    results: Sequence[CallResult],
    summaries: Sequence[ModelSummary],
    skipped: Dict[str, str],
    metadata: Dict[str, Any],
) -> None:
    """Write the raw run data to a JSON file.

    Args:
        output_path: Destination file
        samples: Images used
        results: Every call made
        summaries: Per-model statistics
        skipped: Mapping of skipped model name to reason
        metadata: Run-level settings worth recording

    Raises:
        OSError: If the file cannot be written
    """
    payload = {
        "run": metadata,
        "samples": [s.to_dict() for s in samples],
        "results": [asdict(r) for r in results],
        "summaries": [asdict(s) for s in summaries],
        "skipped": skipped,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRaw results written to {output_path}")


def parse_album_locations(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse ``--album-location`` arguments into a lookup table.

    Args:
        values: Strings of the form ``album-substring=Place Name``

    Returns:
        Mapping of lowercased album substring to place name

    Raises:
        SystemExit: If an argument is not in ``key=value`` form
    """
    mapping: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            print(
                f"ERROR: --album-location expects 'album=Place Name', got {value!r}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        key, place = value.split("=", 1)
        if key.strip() and place.strip():
            mapping[key.strip().lower()] = place.strip()
    return mapping


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Ollama vision models through the real smugVision vision layer. "
            "Read-only: never writes to SmugMug and never modifies your config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models that are not installed are skipped with the exact `ollama pull` "
            "command; nothing is downloaded automatically."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Model names to benchmark (default: every installed vision-capable model)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        metavar="N",
        help=f"Number of photos to test (default: {DEFAULT_SAMPLE_COUNT})",
    )
    parser.add_argument(
        "--album",
        action="append",
        metavar="NAME",
        help="Restrict sampling to albums matching this substring (repeatable)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="DIR",
        help="Sample from this directory instead of the smugVision cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        metavar="DIR",
        help="Image cache root (default: cache.directory from config)",
    )
    parser.add_argument(
        "--album-location",
        action="append",
        metavar="ALBUM=PLACE",
        help=(
            "Supply a place name for albums matching ALBUM. SmugMug strips GPS from "
            "downloaded files, so cached photos rarely carry coordinates. (repeatable)"
        ),
    )
    parser.add_argument("--config", metavar="PATH", help="Path to config.yaml (read-only)")
    parser.add_argument("--endpoint", metavar="URL", help="Ollama endpoint (default: from config)")
    parser.add_argument("--timeout", type=int, metavar="SECONDS", help="Per-request timeout")
    parser.add_argument(
        "--temperature", type=float, metavar="T", help="Sampling temperature (default: from config)"
    )
    parser.add_argument(
        "--max-tokens", type=int, metavar="N", help="Max response tokens (default: from config)"
    )
    parser.add_argument(
        "--no-faces", action="store_true", help="Skip face recognition and supply no person names"
    )
    parser.add_argument(
        "--face-backend",
        metavar="NAME",
        help="Face backend override for this run only, e.g. insightface (config is not modified)",
    )
    parser.add_argument(
        "--keep-warmup",
        action="store_true",
        help="Include the first call per model in the timing statistics",
    )
    parser.add_argument(
        "--no-unload",
        dest="unload",
        action="store_false",
        help="Leave each model resident instead of evicting it after its run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"JSON results file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="Show which photos would be used, then exit without calling any model",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.set_defaults(unload=True)
    return parser


def resolve_models(
    requested: Optional[Sequence[str]], installed: Sequence[str]
) -> Tuple[List[str], Dict[str, str]]:
    """Split the requested models into runnable ones and skips.

    Args:
        requested: Explicitly requested model names, or None to use everything installed
        installed: Vision-capable models Ollama reports

    Returns:
        Tuple of (models to run, mapping of skipped model to reason)
    """
    if not requested:
        return list(installed), {}

    runnable: List[str] = []
    skipped: Dict[str, str] = {}
    lookup = {name.lower(): name for name in installed}
    for name in requested:
        key = name.lower()
        if key in lookup:
            runnable.append(lookup[key])
        elif f"{key}:latest" in lookup:
            runnable.append(lookup[f"{key}:latest"])
        else:
            skipped[name] = f"not installed - run: ollama pull {name}"
    return runnable, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark.

    Args:
        argv: Command-line arguments, or None to read from sys.argv

    Returns:
        Process exit code: 0 on success, 1 when nothing could be benchmarked
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    album_locations = parse_album_locations(args.album_location)
    cache_dir = (
        args.cache_dir or Path(config.get("cache.directory", "~/.smugvision/cache")).expanduser()
    )

    if args.samples < 1:
        print("ERROR: --samples must be at least 1", file=sys.stderr)
        return 1

    selected, remaining = select_samples(cache_dir, args.samples, args.album, args.image_dir)
    if not selected:
        location = args.image_dir or cache_dir
        print(f"ERROR: no cached images found under {location}", file=sys.stderr)
        return 1
    if len(selected) < args.samples:
        print(f"NOTE: only {len(selected)} image(s) available; requested {args.samples}")

    recognizer = None if args.no_faces else build_face_recognizer(config, args.face_backend)
    min_confidence = config.get("face_recognition.min_confidence", 0.25)
    samples = gather_context(selected, recognizer, album_locations, min_confidence)
    samples = ensure_people_sample(samples, remaining, recognizer, album_locations, min_confidence)
    print_samples(samples)

    if args.list_samples:
        return 0

    endpoint = args.endpoint or config.get("vision.endpoint")
    installed = discover_installed_models(endpoint)
    if not installed:
        print(
            f"ERROR: no vision-capable models reported by Ollama at "
            f"{endpoint or 'the default host'}. Is `ollama serve` running?",
            file=sys.stderr,
        )
        return 1

    models, skipped = resolve_models(args.models, installed)
    for name, reason in skipped.items():
        print(f"SKIPPING {name}: {reason}")
    if not models:
        print("ERROR: none of the requested models are installed; nothing to do", file=sys.stderr)
        return 1

    print(
        f"\nBenchmarking {len(models)} model(s) over {len(samples)} image(s): {', '.join(models)}"
    )

    watcher = _ParseModeWatcher()
    vision_logger = logging.getLogger("smugvision.vision.llama")
    vision_logger.addHandler(watcher)
    try:
        results: List[CallResult] = []
        for model_name in models:
            results.extend(benchmark_model(model_name, samples, config, args, watcher))
    finally:
        vision_logger.removeHandler(watcher)

    discard_warm_up = not args.keep_warmup and len(samples) > 1
    summaries = [
        summarize([r for r in results if r.model == name], discard_warm_up) for name in models
    ]

    print_table(summaries, discard_warm_up)
    print_outputs(results, samples)

    try:
        write_results(
            args.output,
            samples,
            results,
            summaries,
            skipped,
            {
                "endpoint": endpoint or "(default)",
                "samples": len(samples),
                "warm_up_discarded": discard_warm_up,
                "structured_output": bool(config.get("vision.structured_output", True)),
                "single_call": bool(config.get("vision.single_call", True)),
                "max_image_dimension": config.get("vision.max_image_dimension", 1568),
                "temperature": (
                    args.temperature
                    if args.temperature is not None
                    else config.get("vision.temperature", 0.4)
                ),
                "max_tokens": (
                    args.max_tokens
                    if args.max_tokens is not None
                    else config.get("vision.max_tokens", 400)
                ),
                "face_backend": (
                    None
                    if recognizer is None
                    else (args.face_backend or config.get("face_recognition.backend", "dlib"))
                ),
            },
        )
    except OSError as e:
        print(f"WARNING: could not write {args.output}: {e}", file=sys.stderr)

    return 0 if any(s.succeeded for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Ollama vision model implementation.

This module provides :class:`LlamaVisionModel`, the single adapter used for every
vision-capable model served by Ollama. The class name is historical - it started as a
Llama 3.2 Vision integration - but it is model agnostic and drives whatever model name
it is given (``qwen3-vl:8b``, ``gemma4:latest``, ``minicpm-v``, ...).
"""

import base64
import json
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ollama
from PIL import Image, ImageOps

# Register HEIF/HEIC support if available
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()

    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# httpx is what the ollama client uses under the hood; it is only needed to classify
# transport errors, so a slim install degrades to string matching instead.
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - httpx ships with ollama
    HTTPX_AVAILABLE = False

from smugvision.vision.base import MetadataResult, VisionModel
from smugvision.vision.exceptions import (
    VisionModelError,
    VisionModelConnectionError,
    VisionModelTimeoutError,
    VisionModelInvalidResponseError,
    VisionModelImageError,
)

logger = logging.getLogger(__name__)

# Accepted JSON keys for each generated field, in preference order. Models vary in what
# they name these, so every place that reads, sniffs for, or salvages a field derives its
# vocabulary from HERE -- previously four hand-written lists had already diverged
# ("summary" and "keyword_tags" parsed normally but were unsalvageable from truncated
# JSON, and "labels" was the reverse).
CAPTION_KEYS: Tuple[str, ...] = ("caption", "description", "text", "summary")
TAG_KEYS: Tuple[str, ...] = ("tags", "keywords", "keyword_tags", "labels")

# Regex alternations built from the tables above, so the salvage patterns can never
# drift from the strict parser's vocabulary.
_CAPTION_KEY_ALT = "|".join(CAPTION_KEYS)
_TAG_KEY_ALT = "|".join(TAG_KEYS)

# Ollama accepts think as a bool, or one of "low" / "medium" / "high", or None.
ThinkSetting = Union[bool, str, None]

# Pillow >= 9.1 moved the resampling filters onto Image.Resampling.
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

# Default number of pixels on the long edge before base64 encoding. Vision models tile
# input to roughly 1024-1568px, so anything larger is wasted bandwidth and memory.
DEFAULT_MAX_IMAGE_DIMENSION = 1568


class LlamaVisionModel(VisionModel):
    """Ollama-backed vision model.

    Handles image loading/downscaling/encoding, prompt construction (including
    location, people and relationship context) and response parsing. The preferred
    entry point is :meth:`generate_metadata`, which encodes the image once and makes a
    single chat request constrained by a JSON schema.

    Attributes:
        model_name: Name of the Ollama model to drive
        endpoint: Ollama API endpoint URL (``None`` uses the client default/OLLAMA_HOST)
        timeout: Request timeout in seconds, applied to the underlying HTTP client
        client: The ``ollama.Client`` instance every request goes through
        think: Value forwarded to Ollama's ``think`` parameter
        keep_alive: How long Ollama should keep the model resident between requests
        single_call: Whether caption and tags come from one request
        structured_output: Whether requests use a JSON-Schema ``format``
        max_image_dimension: Long-edge pixel cap applied before encoding (0/None = off)
        jpeg_quality: JPEG quality used when re-encoding the image
        validate_model: Whether to warn when the model is absent from Ollama's tag list
    """

    def __init__(
        self,
        model_name: str = "llama3.2-vision",
        endpoint: Optional[str] = None,
        timeout: int = 120,
        *,
        think: ThinkSetting = False,
        keep_alive: Optional[Union[str, float]] = "30m",
        single_call: bool = True,
        structured_output: bool = True,
        max_image_dimension: Optional[int] = DEFAULT_MAX_IMAGE_DIMENSION,
        jpeg_quality: int = 85,
        validate_model: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama vision model.

        Args:
            model_name: Name of the Ollama model
            endpoint: Optional Ollama endpoint URL (default: the ollama client default,
                normally http://localhost:11434 or ``$OLLAMA_HOST``)
            timeout: Request timeout in seconds, passed to the HTTP client
            think: Forwarded to Ollama as ``think``. ``False`` disables reasoning,
                ``None`` omits the parameter entirely, and "low"/"medium"/"high"
                select a reasoning budget on models that support it.
            keep_alive: Forwarded to Ollama as ``keep_alive`` so the model stays
                resident between images. ``None`` omits the parameter.
            single_call: Request caption and tags in one round trip. ``False`` selects
                the legacy two-call path.
            structured_output: Constrain responses with a JSON schema. ``False``
                selects the legacy free-text prompt plus heuristic parsing.
            max_image_dimension: Downscale the image's long edge to this many pixels
                before encoding. ``0`` or ``None`` disables downscaling. Images are
                never upscaled.
            jpeg_quality: JPEG quality (1-95) used when re-encoding for transport
            validate_model: Warn (never fail) when the model name is missing from
                Ollama's tag list
            **kwargs: Ignored, logged at debug level. Present so that forwarding a
                whole config block cannot break construction.

        Raises:
            VisionModelConnectionError: If unable to connect to Ollama
        """
        super().__init__(model_name, endpoint)
        self.timeout = timeout
        self.think: ThinkSetting = think
        self.keep_alive = keep_alive
        self.single_call = bool(single_call)
        self.structured_output = bool(structured_output)
        self.max_image_dimension = self._normalize_dimension(max_image_dimension)
        self.jpeg_quality = max(1, min(int(jpeg_quality), 95))
        self.validate_model = bool(validate_model)

        # Flipped to False if the server rejects the think parameter for this model.
        self._think_supported = True

        if kwargs:
            logger.debug(f"Ignoring unsupported LlamaVisionModel options: {sorted(kwargs)}")

        # A real client bound to the requested endpoint. host=None lets the ollama
        # client fall back to $OLLAMA_HOST / http://localhost:11434.
        client_kwargs: Dict[str, Any] = {}
        if timeout:
            # ollama.Client forwards unknown kwargs to httpx, where timeout is honored.
            client_kwargs["timeout"] = timeout
        try:
            self.client = ollama.Client(host=endpoint, **client_kwargs)
        except Exception as e:
            raise VisionModelConnectionError(
                f"Failed to create Ollama client for endpoint {endpoint or '(default)'}: {e}"
            ) from e
        logger.debug(
            f"Ollama client bound to {endpoint or '(default host)'} "
            f"(timeout={timeout}s, keep_alive={keep_alive}, think={think}, "
            f"single_call={self.single_call}, structured_output={self.structured_output}, "
            f"max_image_dimension={self.max_image_dimension}, jpeg_quality={self.jpeg_quality})"
        )

        # Verify connection to Ollama
        try:
            self._verify_connection()
        except VisionModelConnectionError:
            raise
        except Exception as e:
            raise VisionModelConnectionError(f"Failed to connect to Ollama service: {e}") from e

    @staticmethod
    def _normalize_dimension(value: Optional[Union[int, str]]) -> Optional[int]:
        """Coerce a configured max-dimension value into a positive int or None.

        Args:
            value: Raw configuration value (int, numeric string, 0, or None)

        Returns:
            A positive pixel count, or None when downscaling is disabled
        """
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid max_image_dimension {value!r}; downscaling disabled")
            return None
        return parsed if parsed > 0 else None

    def _verify_connection(self) -> None:
        """Verify connection to the Ollama service and sanity-check the model name.

        The model-name check is advisory only: Ollama can serve a model that is not in
        the tag list, so a mismatch logs a warning instead of raising.

        Raises:
            VisionModelConnectionError: If the service cannot be reached
        """
        try:
            models_response = self.client.list()
        except Exception as e:
            where = self.endpoint or "the default host ($OLLAMA_HOST or http://localhost:11434)"
            raise VisionModelConnectionError(
                f"Cannot connect to Ollama service at {where}. Make sure Ollama is running "
                f"and that vision.endpoint / $OLLAMA_HOST point at it: {e}"
            ) from e

        model_names = self._extract_model_names(models_response)
        logger.debug(f"Connected to Ollama. Available models: {model_names}")

        if not self.validate_model:
            return

        if not model_names:
            # Empty model list - model might still work, just log at debug level
            logger.debug(
                f"Ollama model list is empty, but model '{self.model_name}' "
                f"may still be available and work."
            )
            return

        if self.model_name in model_names:
            return

        # Check for partial matches (e.g., "llama3.2-vision" vs "llama3.2-vision:latest")
        for available_model in model_names:
            if self.model_name in available_model or available_model in self.model_name:
                logger.debug(
                    f"Model '{self.model_name}' matches available model: {available_model}"
                )
                return

        logger.warning(
            f"Model '{self.model_name}' not found in Ollama model list. "
            f"Available models: {model_names}. "
            f"The model may still work if it's available. "
            f"If not, run: ollama pull {self.model_name}"
        )

    @staticmethod
    def _extract_model_names(models_response: Any) -> List[str]:
        """Pull model names out of an Ollama list response.

        Handles the pydantic ``ListResponse`` returned by modern clients as well as the
        plain dict/list shapes older versions produced.

        Args:
            models_response: Whatever ``Client.list()`` returned

        Returns:
            List of model name strings (possibly empty)
        """
        models_list: Any = []
        if models_response is None:
            models_list = []
        elif isinstance(models_response, dict):
            models_list = models_response.get("models", [])
        elif isinstance(models_response, list):
            models_list = models_response
        else:
            models_list = getattr(models_response, "models", []) or []

        model_names: List[str] = []
        for model in models_list:
            name = ""
            if isinstance(model, str):
                name = model
            elif isinstance(model, dict):
                name = model.get("name") or model.get("model") or ""
            else:
                name = getattr(model, "model", "") or getattr(model, "name", "") or ""
            if name:
                model_names.append(str(name))
        return model_names

    def _strip_thinking_tags(self, content: str) -> str:
        """Strip <think>...</think> tags and other reasoning blocks from content.

        Some models (like MiniCPM-V 4.5) include their reasoning process in the
        response wrapped in <think> tags. This method removes those blocks and
        returns only the actual response. It remains part of the legacy free-text
        path and is applied to every response for safety.

        Args:
            content: Raw content that may contain thinking tags

        Returns:
            Cleaned content with thinking blocks removed
        """
        if not content:
            return content

        # Remove <think>...</think> blocks (including multiline)
        # Use DOTALL flag so . matches newlines
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Also handle unclosed <think> tags (model may have been cut off)
        # Remove everything from <think> to end if no closing tag
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Clean up any leftover whitespace
        cleaned = cleaned.strip()

        # If we removed everything, the actual content might be after the think block
        # or the model only produced thinking - log this
        if content and not cleaned:
            logger.warning(
                "Content was entirely within <think> tags. "
                "Set vision.think to false or adjust the prompt to disable thinking mode."
            )

        return cleaned

    def _encode_image(
        self,
        image_path: str,
        max_dimension: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> str:
        """Encode an image file to a base64 JPEG string.

        Supports various image formats including JPEG, PNG, and HEIC/HEIF. The image is
        rotated per its EXIF orientation, downscaled so its long edge is at most
        ``max_dimension`` pixels (never upscaled) and re-encoded as JPEG.

        Args:
            image_path: Path to the image file
            max_dimension: Long-edge pixel cap. ``None`` uses the instance setting;
                ``0`` explicitly disables downscaling for this call.
            jpeg_quality: JPEG quality; ``None`` uses the instance setting

        Returns:
            Base64-encoded image string

        Raises:
            VisionModelImageError: If image cannot be read or encoded
        """
        if max_dimension is None:
            limit = self.max_image_dimension
        else:
            limit = self._normalize_dimension(max_dimension)
        quality = max(1, min(int(jpeg_quality), 95)) if jpeg_quality else self.jpeg_quality

        try:
            image_file = Path(image_path)
            if not image_file.exists():
                raise VisionModelImageError(f"Image file not found: {image_path}")

            # Check for HEIC format
            file_ext = image_file.suffix.lower()
            is_heic = file_ext in [".heic", ".heif"]

            if is_heic and not HEIC_SUPPORT:
                raise VisionModelImageError(
                    "HEIC format not supported. Install pillow-heif: pip install pillow-heif"
                )

            # Validate and open image
            try:
                # First verify the image is valid
                with Image.open(image_path) as probe:
                    probe.verify()

                # Reopen for actual reading (verify() closes the image)
                img: Image.Image = Image.open(image_path)
                try:
                    # Honor EXIF orientation so portrait shots are not sent sideways
                    # (the JPEG we produce below carries no EXIF block).
                    img = ImageOps.exif_transpose(img) or img

                    # JPEG cannot store alpha or exotic modes; flatten to RGB.
                    if img.mode not in ("RGB", "L"):
                        logger.debug(f"Converting image from {img.mode} to RGB")
                        if img.mode in ("RGBA", "LA", "PA"):
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            alpha = img.convert("RGBA").split()[-1]
                            background.paste(img.convert("RGB"), mask=alpha)
                            img.close()
                            img = background
                        else:
                            converted = img.convert("RGB")
                            img.close()
                            img = converted

                    original_size = img.size
                    img = self._downscale(img, limit)
                    if img.size != original_size:
                        logger.debug(
                            f"Downscaled image from {original_size[0]}x{original_size[1]} "
                            f"to {img.size[0]}x{img.size[1]} (max {limit}px)"
                        )

                    # Save to bytes in JPEG format for encoding
                    output = BytesIO()
                    img.save(output, format="JPEG", quality=quality)
                    image_data = output.getvalue()
                    output.close()
                finally:
                    img.close()

            except VisionModelImageError:
                raise
            except Exception as e:
                error_msg = f"Invalid image format for {image_path}: {e}"
                if is_heic:
                    error_msg += (
                        "\nNote: HEIC support requires pillow-heif. "
                        "Install with: pip install pillow-heif"
                    )
                raise VisionModelImageError(error_msg) from e

            encoded = base64.b64encode(image_data).decode("utf-8")
            logger.debug(
                f"Encoded image {image_path} ({len(image_data)} bytes, format: {file_ext})"
            )
            return encoded

        except VisionModelImageError:
            raise
        except Exception as e:
            raise VisionModelImageError(f"Failed to encode image {image_path}: {e}") from e

    @staticmethod
    def _downscale(img: Image.Image, max_dimension: Optional[int]) -> Image.Image:
        """Downscale an image so its long edge fits within max_dimension.

        Aspect ratio is preserved and images smaller than the limit are returned
        untouched - this never upscales.

        Args:
            img: Source PIL image
            max_dimension: Long-edge pixel cap, or None to disable

        Returns:
            The original image, or a resized copy (the original is closed)
        """
        if not max_dimension:
            return img

        width, height = img.size
        longest = max(width, height)
        if longest <= max_dimension:
            return img

        scale = max_dimension / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = img.resize(new_size, LANCZOS)
        img.close()
        return resized

    def _wrap_call_error(self, error: Exception) -> VisionModelError:
        """Translate a transport/server error into the right vision exception.

        Args:
            error: The exception raised by the Ollama client

        Returns:
            A VisionModelError subclass describing the failure
        """
        message = str(error)
        lowered = message.lower()

        is_timeout = isinstance(error, TimeoutError)
        is_connection = False
        if HTTPX_AVAILABLE:
            is_timeout = is_timeout or isinstance(error, httpx.TimeoutException)
            is_connection = isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))

        if is_timeout or "timed out" in lowered or "timeout" in lowered:
            return VisionModelTimeoutError(
                f"Request to {self.model_name} timed out after {self.timeout}s: {message}"
            )
        if is_connection or "connection" in lowered:
            return VisionModelConnectionError(
                f"Cannot reach Ollama at {self.endpoint or '(default host)'}: {message}"
            )
        return VisionModelError(f"Error calling Ollama model {self.model_name}: {message}")

    @staticmethod
    def _is_think_unsupported(error: Exception) -> bool:
        """Detect an Ollama error caused by requesting thinking on a model without it.

        Args:
            error: The exception raised by the Ollama client

        Returns:
            True when retrying without the ``think`` parameter is worth attempting
        """
        lowered = str(error).lower()
        if isinstance(error, TypeError) and "think" in lowered:
            # Client-side rejection: an ollama-python older than 0.5.0 has no `think`
            # parameter, so Client.chat raises TypeError before any request is sent.
            return True
        return "think" in lowered and (
            "support" in lowered or "unknown" in lowered or "invalid" in lowered
        )

    def _call_ollama(
        self,
        prompt: str,
        image_base64: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call the Ollama chat API with an image and prompt.

        Args:
            prompt: Text prompt for the model
            image_base64: Base64-encoded image
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            response_format: Optional JSON Schema passed as Ollama's ``format``

        Returns:
            Model response text (thinking blocks stripped)

        Raises:
            VisionModelTimeoutError: If the request times out
            VisionModelConnectionError: If Ollama cannot be reached
            VisionModelInvalidResponseError: If the response has no usable content
            VisionModelError: For other errors
        """
        messages = [{"role": "user", "content": prompt, "images": [image_base64]}]

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if response_format is not None:
            call_kwargs["format"] = response_format
        if self.keep_alive is not None:
            call_kwargs["keep_alive"] = self.keep_alive
        if self.think is not None and self._think_supported:
            call_kwargs["think"] = self.think

        logger.debug(
            f"Sending request to {self.model_name} (temperature={temperature}, "
            f"max_tokens={max_tokens}, structured={response_format is not None}, "
            f"think={call_kwargs.get('think', '(omitted)')}, "
            f"keep_alive={call_kwargs.get('keep_alive', '(omitted)')})"
        )

        start_time = time.time()
        try:
            response = self.client.chat(**call_kwargs)
        except Exception as e:
            if "think" in call_kwargs and self._is_think_unsupported(e):
                if isinstance(e, TypeError):
                    logger.warning(
                        f"The installed ollama-python client does not accept the 'think' "
                        f"parameter ({e}); retrying without it. Upgrade to ollama>=0.5.0 - "
                        f"until then vision.think cannot be honoured, and a reasoning model "
                        f"can spend the whole token budget before emitting any content."
                    )
                else:
                    logger.warning(
                        f"Model {self.model_name} rejected the 'think' parameter ({e}); "
                        f"retrying without it and disabling it for this session."
                    )
                self._think_supported = False
                call_kwargs.pop("think", None)
                try:
                    response = self.client.chat(**call_kwargs)
                except Exception as retry_error:
                    raise self._wrap_call_error(retry_error) from retry_error
            else:
                raise self._wrap_call_error(e) from e

        elapsed = time.time() - start_time
        logger.debug(f"Received response in {elapsed:.2f}s")

        content = self._extract_content(response)
        content = self._strip_thinking_tags(content)

        if not content:
            if self.think is not None and not self._think_supported:
                hint = (
                    "The installed ollama-python client rejected 'think', so reasoning "
                    "cannot be switched off - upgrade to ollama>=0.5.0, or raise "
                    "vision.max_tokens."
                )
            else:
                hint = (
                    "If this is a reasoning model, set vision.think to false or raise "
                    "vision.max_tokens."
                )
            raise VisionModelInvalidResponseError(
                f"Empty response content from {self.model_name}. {hint}"
            )

        return content

    def _extract_content(self, response: Any) -> str:
        """Extract the assistant message content from an Ollama chat response.

        Args:
            response: Raw response object or dict from ``Client.chat``

        Returns:
            The message content, stripped (may be empty)

        Raises:
            VisionModelInvalidResponseError: If the response has no message at all
        """
        if response is None:
            raise VisionModelInvalidResponseError("No response returned from Ollama")

        message: Any = None
        if isinstance(response, dict):
            message = response.get("message")
        else:
            message = getattr(response, "message", None)

        if message is None:
            raise VisionModelInvalidResponseError(
                f"Invalid response structure from Ollama: {response}"
            )

        content = ""
        if isinstance(message, dict):
            content = message.get("content") or ""
        else:
            content = getattr(message, "content", "") or ""

        content = str(content).strip()

        if not content:
            thinking = (
                message.get("thinking")
                if isinstance(message, dict)
                else getattr(message, "thinking", None)
            )
            if thinking:
                logger.warning(
                    f"{self.model_name} returned reasoning but no answer content. "
                    f"Set vision.think to false so the model answers directly."
                )

        return content

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_combined_prompt(
        self,
        caption_instruction: str,
        tags_instruction: str,
        structured: bool,
    ) -> str:
        """Build the instruction half of a single-call prompt.

        Args:
            caption_instruction: Caption instruction (empty to skip captions)
            tags_instruction: Tags instruction (empty to skip tags)
            structured: Whether a JSON schema will constrain the response

        Returns:
            The combined instruction text, without context (added separately)
        """
        want_caption = bool(caption_instruction and caption_instruction.strip())
        want_tags = bool(tags_instruction and tags_instruction.strip())

        parts: List[str] = []
        if want_caption and want_tags:
            parts.append("Analyze this image and produce both a caption and keyword tags.")
            parts.append(f"CAPTION INSTRUCTIONS: {caption_instruction.strip()}")
            parts.append(f"TAG INSTRUCTIONS: {tags_instruction.strip()}")
            if structured:
                parts.append(
                    'Respond with a single JSON object: {"caption": "<the caption>", '
                    '"tags": ["tag1", "tag2"]}. Use lowercase single words or short '
                    "phrases for tags."
                )
            else:
                parts.append(
                    "Respond in exactly this format and nothing else:\n"
                    "CAPTION: <the caption>\n"
                    "TAGS: <comma-separated tags>"
                )
        elif want_caption:
            parts.append(caption_instruction.strip())
            if structured:
                parts.append('Respond with a single JSON object: {"caption": "<the caption>"}.')
        elif want_tags:
            parts.append(tags_instruction.strip())
            if structured:
                parts.append('Respond with a single JSON object: {"tags": ["tag1", "tag2"]}.')

        return "\n\n".join(parts)

    @staticmethod
    def _join_names(names: List[str], *, oxford: bool = True) -> str:
        """Join person names into a natural-language list.

        Args:
            names: Display names, already underscore-converted
            oxford: Whether three-or-more names take an Oxford comma before "and".
                The relationship wording has historically omitted it.

        Returns:
            "Ann", "Ann and Bob", or "Ann, Bob, and Cid" / "Ann, Bob and Cid"
        """
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        tail = f", and {names[-1]}" if oxford else f" and {names[-1]}"
        return ", ".join(names[:-1]) + tail

    @staticmethod
    def _clean_album_name(album_name: str) -> str:
        """Strip SmugMug date prefixes from an album name before prompting.

        SmugMug albums are commonly titled with a leading date, e.g.
        ``2025:03:26 Grand Finale Cleaning House``. Passing that raw into the
        prompt leads models to echo the timestamp verbatim into the caption,
        so the prefix is removed and only the human-readable title is kept.

        Handles ``YYYY:MM:DD``, ``YYYY-MM-DD`` and ``YYYY_MM_DD``, optionally
        followed by a separator. A name that is *only* a date is returned
        unchanged, since stripping it would leave no context at all.

        Args:
            album_name: Raw album name as returned by the SmugMug API

        Returns:
            The album name with any leading date prefix removed, stripped of
            surrounding whitespace
        """
        cleaned = re.sub(
            r"^\s*\d{4}[:\-_]\d{2}[:\-_]\d{2}\s*[-–—:_]?\s*",
            "",
            album_name,
        ).strip()
        return cleaned or album_name.strip()

    def _enhance_prompt_with_context(
        self,
        prompt: str,
        location_context: Optional[str],
        person_names: Optional[List[str]] = None,
        total_faces: Optional[int] = None,
        album_name: Optional[str] = None,
    ) -> str:
        """Enhance a prompt with album, location, people and relationship context.

        Context is expected to arrive as arguments rather than pre-baked into
        ``prompt``; passing it both ways sends the model duplicate information.

        Args:
            prompt: Original prompt text
            location_context: Optional resolved place name
            person_names: Optional list of recognized person names. Raw
                reference-folder names (``John_Doe``) are preferred, since relationship
                lookups match on them; underscores are formatted out for the model.
            total_faces: Optional count of faces detected, which may exceed the number
                of recognized names
            album_name: Optional album name

        Returns:
            Enhanced prompt with context naturally incorporated
        """
        context_parts = []

        # Add album context if available
        if album_name:
            clean_album = self._clean_album_name(album_name)
            if clean_album:
                context_parts.append(f"This photo is from an album titled '{clean_album}'.")

        # Add person names with relationship context if available
        if person_names:
            # Try to load relationship context (~/.smugvision/relationships.yaml)
            try:
                from smugvision.utils.relationships import get_relationship_manager

                rel_manager = get_relationship_manager()
                relationship_context = rel_manager.generate_context(person_names)
            except Exception as e:
                logger.debug(f"Could not load relationship context: {e}")
                relationship_context = None
            # Format names: replace underscores with spaces
            formatted_names = [name.replace("_", " ") for name in person_names]
            recognized_count = len(formatted_names)

            # Build person context - use relationship description if available.
            # The 1 / 2 / 3+ x with/without-total_faces matrix differs only in the
            # subject clause and whether names are joined with an Oxford comma, so it
            # is expressed as substitutions rather than six near-identical blocks.
            plural = "" if recognized_count == 1 else "s"
            # The single-person wordings repeat the name as a parenthetical hint.
            name_hint = f" ({formatted_names[0]})" if recognized_count == 1 else ""

            if relationship_context:
                # Relationship wording has always joined without the Oxford comma.
                names_str = self._join_names(formatted_names, oxford=False)
                context_parts.append(
                    f"The people in this image are {names_str} ({relationship_context}). "
                    f"Please use their names and incorporate the relationship information "
                    f"naturally into your description."
                )
            elif total_faces and total_faces > recognized_count:
                # Some people in the photo could not be identified.
                names_str = self._join_names(formatted_names)
                qualifier = {1: "One of them is", 2: "Two of them are"}.get(
                    recognized_count, "Some of them are"
                )
                context_parts.append(
                    f"There are {total_faces} people in this image. "
                    f"{qualifier} {names_str}. "
                    f"Please use their name{plural}{name_hint} when describing them, "
                    f"and mention that there are other people present."
                )
            else:
                # All faces were recognized (or total_faces not provided).
                names_str = self._join_names(formatted_names)
                subject = (
                    "The person in this image is"
                    if recognized_count == 1
                    else "The people in this image are"
                )
                context_parts.append(
                    f"{subject} {names_str}. "
                    f"Please use their name{plural}{name_hint} when describing them."
                )
        elif total_faces and total_faces > 0:
            # Faces were detected but nobody was recognized - still useful context.
            people_word = "person" if total_faces == 1 else "people"
            context_parts.append(
                f"There {'is' if total_faces == 1 else 'are'} {total_faces} {people_word} "
                f"visible in this image, none of whom could be identified by name. "
                f"Do not invent names."
            )

        # Add location context if available
        if location_context:
            context_parts.append(f"This image was taken at {location_context}.")

        if not context_parts:
            return prompt

        # Combine all context
        context_text = " ".join(context_parts)

        # Check if prompt already mentions location or people
        has_context_mention = any(
            keyword.lower() in prompt.lower()
            for keyword in [
                "location",
                "where",
                "place",
                "taken",
                "exif",
                "person",
                "people",
                "who",
            ]
        )

        if has_context_mention:
            # If prompt already mentions context, append it
            enhanced = (
                f"{prompt}\n\nAdditional context: {context_text} "
                f"Please incorporate this information naturally."
            )
        else:
            # Otherwise, add it as context with explicit instruction
            enhanced = (
                f"{prompt}\n\nContext: {context_text} "
                f"Please incorporate this information naturally into your description, "
                f"including using the person's name when referring to them."
            )

        logger.debug(
            f"Enhanced prompt with context: album={album_name}, location={location_context}, "
            f"people={person_names}, total_faces={total_faces}"
        )
        return enhanced

    # ------------------------------------------------------------------
    # Structured response handling
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schema(want_caption: bool, want_tags: bool) -> Dict[str, Any]:
        """Build the JSON Schema passed to Ollama as ``format``.

        Args:
            want_caption: Whether a caption field is requested
            want_tags: Whether a tags field is requested

        Returns:
            A JSON Schema dict describing the expected object
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []
        if want_caption:
            properties["caption"] = {"type": "string"}
            required.append("caption")
        if want_tags:
            properties["tags"] = {"type": "array", "items": {"type": "string"}}
            required.append("tags")
        return {"type": "object", "properties": properties, "required": required}

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        """Strip a surrounding markdown code fence (```json ... ```) from a response.

        Args:
            content: Raw response text

        Returns:
            The fenced body when the text is fenced, otherwise the stripped input
        """
        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        return text

    @classmethod
    def _extract_json_text(cls, content: str) -> str:
        """Isolate a JSON object from a response that may wrap it in prose or fences.

        Args:
            content: Raw response text

        Returns:
            The most likely JSON payload, or the input unchanged
        """
        text = cls._strip_code_fence(content)

        if text.startswith("{") or text.startswith("["):
            return text

        # Fall back to the first balanced-looking object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            stop = end + 1
            return text[start:stop]
        return text

    @classmethod
    def _looks_like_json(cls, content: str) -> bool:
        """Report whether a response is JSON debris rather than prose.

        Used to keep the free-text parser away from broken JSON: running it on
        ``{"tags": ["a"`` would happily emit ``tags": ["a`` as a caption or tag.

        The whole reply must read as JSON - it must start with a brace/bracket, or
        carry a quoted "caption"/"tags" key. Prose that merely *contains* braces
        (a caption about a whiteboard, a receipt, or a code screenshot) is prose,
        and must stay reachable by the free-text parser.

        Args:
            content: Raw response text

        Returns:
            True when the text is structured output rather than prose
        """
        text = cls._strip_code_fence(content)
        if text.startswith("{") or text.startswith("["):
            return True
        return bool(re.search(r'"(?:caption|tags|keywords|description)"\s*:', text, re.IGNORECASE))

    @staticmethod
    def _unescape_json_string(value: str) -> str:
        """Decode a raw JSON string body (the part between the quotes).

        Args:
            value: The escaped string contents

        Returns:
            The decoded text, or the input unchanged if it cannot be decoded
        """
        try:
            decoded = json.loads(f'"{value}"')
            return decoded if isinstance(decoded, str) else value
        except ValueError:
            return value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")

    def _salvage_json_fields(
        self, content: str, want_caption: bool, want_tags: bool
    ) -> Tuple[str, List[str]]:
        """Regex-recover caption/tags from malformed or truncated JSON.

        Constrained decoding can be cut off mid-object when the token budget runs out,
        which leaves valid data inside invalid JSON. This pulls out whatever survived.

        Args:
            content: Raw response text
            want_caption: Whether a caption is expected
            want_tags: Whether tags are expected

        Returns:
            Tuple of (caption, tags); either may be empty when nothing was recoverable
        """
        caption = ""
        tags: List[str] = []

        if want_caption:
            match = re.search(
                rf'"(?:{_CAPTION_KEY_ALT})"\s*:\s*"((?:[^"\\]|\\.)*)"',
                content,
                flags=re.IGNORECASE,
            )
            if not match:
                # Truncated mid-caption: no closing quote to match against.
                match = re.search(
                    rf'"(?:{_CAPTION_KEY_ALT})"\s*:\s*"((?:[^"\\]|\\.)*)$',
                    content,
                    flags=re.IGNORECASE,
                )
            if match:
                caption = self._unescape_json_string(match.group(1)).strip()

        if want_tags:
            match = re.search(
                rf'"(?:{_TAG_KEY_ALT})"\s*:\s*\[(.*?)(?:\]|$)',
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                items = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))
                tags = self._coerce_tag_list([self._unescape_json_string(i) for i in items])

        return caption, tags

    @staticmethod
    def _coerce_caption(value: Any) -> str:
        """Coerce a model-supplied caption value into a plain string.

        Args:
            value: Whatever the model put in the caption field

        Returns:
            A stripped caption string (possibly empty)
        """
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            parts = [str(v).strip() for v in value if str(v).strip()]
            return " ".join(parts).strip()
        if isinstance(value, dict):
            for key in ("caption", "text", "value", "description"):
                if key in value:
                    return LlamaVisionModel._coerce_caption(value[key])
            return ""
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _coerce_tag_list(value: Any) -> List[str]:
        """Coerce a model-supplied tags value into a list of strings.

        Handles proper lists, comma/newline separated strings, and dicts keyed by tag.

        Args:
            value: Whatever the model put in the tags field

        Returns:
            List of non-empty tag strings
        """
        raw_items: List[Any]
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = re.split(r"[,;\n]+", value)
        elif isinstance(value, dict):
            # Some models emit {"tags": {"1": "dog", "2": "park"}} or {"dog": true}
            values = list(value.values())
            if values and all(isinstance(v, str) for v in values):
                raw_items = values
            else:
                raw_items = list(value.keys())
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]

        tags: List[str] = []
        seen = set()
        for item in raw_items:
            if isinstance(item, dict):
                text = LlamaVisionModel._coerce_caption(item)
            elif isinstance(item, (list, tuple)):
                text = " ".join(str(v) for v in item)
            else:
                text = str(item)
            text = " ".join(text.strip().strip("\"'`.,;:!?-").split())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            tags.append(text)
        return tags

    def _parse_structured_response(
        self, content: str, want_caption: bool, want_tags: bool
    ) -> Tuple[str, List[str]]:
        """Parse a JSON response, coercing loose shapes into (caption, tags).

        Args:
            content: Raw response text
            want_caption: Whether a caption is expected
            want_tags: Whether tags are expected

        Returns:
            Tuple of (caption, tags)

        Raises:
            VisionModelInvalidResponseError: If the JSON cannot be parsed, or the shape
                does not contain the requested fields
        """
        try:
            data = json.loads(self._extract_json_text(content))
        except (ValueError, TypeError) as e:
            raise VisionModelInvalidResponseError(f"Response was not valid JSON: {e}") from e

        # Unwrap a single wrapper key, e.g. {"result": {"caption": ..., "tags": [...]}}
        for _ in range(3):
            if (
                isinstance(data, dict)
                and "caption" not in data
                and "tags" not in data
                and len(data) == 1
            ):
                inner = next(iter(data.values()))
                if isinstance(inner, (dict, list)):
                    data = inner
                    continue
            break

        caption = ""
        tags: List[str] = []

        if isinstance(data, dict):
            lowered = {str(k).lower(): v for k, v in data.items()}
            if want_caption:
                for key in CAPTION_KEYS:
                    if key in lowered:
                        caption = self._coerce_caption(lowered[key])
                        if caption:
                            break
            if want_tags:
                for key in TAG_KEYS:
                    if key in lowered:
                        tags = self._coerce_tag_list(lowered[key])
                        if tags:
                            break
        elif isinstance(data, list):
            # A bare list is only meaningful as tags
            if want_tags and not want_caption:
                tags = self._coerce_tag_list(data)
        elif isinstance(data, str):
            if want_caption and not want_tags:
                caption = data.strip()

        if want_caption and not caption:
            raise VisionModelInvalidResponseError(
                f"Structured response from {self.model_name} had no usable caption: "
                f"{content[:200]}"
            )
        if want_tags and not tags:
            raise VisionModelInvalidResponseError(
                f"Structured response from {self.model_name} had no usable tags: {content[:200]}"
            )

        return caption, tags

    def _parse_freetext_response(
        self, content: str, want_caption: bool, want_tags: bool
    ) -> Tuple[str, List[str]]:
        """Parse a free-text response into (caption, tags) - the legacy path.

        Understands the ``CAPTION:`` / ``TAGS:`` convention used by the free-text
        single-call prompt, and degrades to treating the whole response as a caption
        and/or running it through :meth:`_parse_tags`.

        Args:
            content: Raw response text
            want_caption: Whether a caption is expected
            want_tags: Whether tags are expected

        Returns:
            Tuple of (caption, tags)
        """
        text = content.strip()
        caption = ""
        tags: List[str] = []

        caption_match = re.search(
            r"caption\s*[:\-]\s*(.+?)(?=\n\s*(?:tags|keywords)\s*[:\-]|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        tags_match = re.search(
            r"(?:tags|keywords)\s*[:\-]\s*(.+)\Z", text, flags=re.IGNORECASE | re.DOTALL
        )

        if want_caption:
            if caption_match:
                caption = caption_match.group(1).strip().strip("\"'")
            elif want_tags and tags_match:
                # Everything before the tags block is the caption
                caption = text[: tags_match.start()].strip().strip("\"'")
            else:
                caption = text.strip().strip("\"'")

        if want_tags:
            if tags_match:
                tags = self._parse_tags(tags_match.group(1))
            elif want_caption and caption_match:
                # Tags may follow the caption block without a label
                tail_start = caption_match.end()
                remainder = text[tail_start:].strip()
                tags = self._parse_tags(remainder) if remainder else []
            else:
                tags = self._parse_tags(text)

        return caption, tags

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_metadata(
        self,
        image_path: str,
        caption_instruction: str,
        tags_instruction: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 400,
        location_context: Optional[str] = None,
        person_names: Optional[List[str]] = None,
        total_faces: Optional[int] = None,
        album_name: Optional[str] = None,
    ) -> MetadataResult:
        """Generate a caption and keyword tags for an image.

        By default this encodes the image once and makes a single chat request whose
        response is constrained by a JSON schema. Two escape hatches exist for models
        that handle schemas or combined prompts badly:

        * ``structured_output=False`` - free-text response plus heuristic parsing.
        * ``single_call=False`` - one request for the caption and one for the tags,
          both receiving the same context enrichment.

        If a structured response fails to parse, the raw text is run through the legacy
        free-text parser rather than failing the image.

        Args:
            image_path: Path to the image file
            caption_instruction: Instruction for the caption. Empty skips captions.
            tags_instruction: Instruction for the tags. Empty skips tags.
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response, shared by caption and tags
            location_context: Optional resolved place name
            person_names: Optional recognized person names (raw ``John_Doe`` form)
            total_faces: Optional count of detected faces, which may exceed the number
                of recognized names
            album_name: Optional album name

        Returns:
            MetadataResult with the caption, tags, model name and elapsed time

        Raises:
            VisionModelError: If generation fails
            VisionModelImageError: If the image cannot be read or encoded
        """
        want_caption = bool(caption_instruction and caption_instruction.strip())
        want_tags = bool(tags_instruction and tags_instruction.strip())

        if not want_caption and not want_tags:
            raise VisionModelError(
                "generate_metadata requires at least one of caption_instruction or "
                "tags_instruction to be non-empty"
            )

        start_time = time.time()

        try:
            # ONE encode, reused by every request below.
            image_base64 = self._encode_image(image_path)

            if self.single_call or not (want_caption and want_tags):
                caption, tags = self._generate_combined(
                    image_base64=image_base64,
                    caption_instruction=caption_instruction,
                    tags_instruction=tags_instruction,
                    want_caption=want_caption,
                    want_tags=want_tags,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    location_context=location_context,
                    person_names=person_names,
                    total_faces=total_faces,
                    album_name=album_name,
                )
            else:
                caption, tags = self._generate_two_calls(
                    image_base64=image_base64,
                    caption_instruction=caption_instruction,
                    tags_instruction=tags_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    location_context=location_context,
                    person_names=person_names,
                    total_faces=total_faces,
                    album_name=album_name,
                )
        except VisionModelError:
            raise
        except Exception as e:
            raise VisionModelError(f"Failed to generate metadata for {image_path}: {e}") from e

        processing_time = time.time() - start_time
        logger.debug(
            f"generate_metadata({Path(image_path).name}) produced "
            f"{len(caption)} caption chars and {len(tags)} tags in {processing_time:.2f}s"
        )

        return MetadataResult(
            caption=caption,
            tags=tags,
            confidence=1.0,
            model_used=self.model_name,
            processing_time=processing_time,
        )

    def _generate_combined(
        self,
        *,
        image_base64: str,
        caption_instruction: str,
        tags_instruction: str,
        want_caption: bool,
        want_tags: bool,
        temperature: float,
        max_tokens: int,
        location_context: Optional[str],
        person_names: Optional[List[str]],
        total_faces: Optional[int],
        album_name: Optional[str],
    ) -> Tuple[str, List[str]]:
        """Run one chat request covering everything that was requested.

        Args:
            image_base64: Pre-encoded image
            caption_instruction: Caption instruction (may be empty)
            tags_instruction: Tags instruction (may be empty)
            want_caption: Whether a caption is requested
            want_tags: Whether tags are requested
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            location_context: Optional resolved place name
            person_names: Optional recognized person names
            total_faces: Optional detected face count
            album_name: Optional album name

        Returns:
            Tuple of (caption, tags)
        """
        prompt = self._build_combined_prompt(
            caption_instruction, tags_instruction, structured=self.structured_output
        )
        prompt = self._enhance_prompt_with_context(
            prompt, location_context, person_names, total_faces, album_name
        )
        self._log_prompt("metadata", prompt)

        schema = self._build_schema(want_caption, want_tags) if self.structured_output else None
        content = self._call_ollama(
            prompt, image_base64, temperature, max_tokens, response_format=schema
        )

        return self._interpret_response(content, want_caption, want_tags, structured=schema)

    def _interpret_response(
        self,
        content: str,
        want_caption: bool,
        want_tags: bool,
        structured: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[str]]:
        """Turn a raw response into (caption, tags), recovering from bad JSON.

        Recovery order: strict JSON parse, then a regex salvage of the JSON fields
        (which handles responses truncated by the token budget), then - only when the
        response is prose rather than JSON debris - the legacy free-text parser.

        Args:
            content: Raw response text (non-empty)
            want_caption: Whether a caption is expected
            want_tags: Whether tags are expected
            structured: The schema that was requested, or None for the free-text path

        Returns:
            Tuple of (caption, tags)

        Raises:
            VisionModelInvalidResponseError: If nothing usable could be recovered
        """
        caption = ""
        tags: List[str] = []
        looks_json = self._looks_like_json(content)
        recovered = False

        # Try JSON whenever a schema was requested, and also when a free-text response
        # happens to come back as JSON anyway.
        if structured is not None or looks_json:
            try:
                caption, tags = self._parse_structured_response(content, want_caption, want_tags)
                recovered = True
            except VisionModelInvalidResponseError as e:
                logger.warning(
                    f"Structured output from {self.model_name} was unusable ({e}); "
                    f"attempting recovery."
                )
                caption, tags = self._salvage_json_fields(content, want_caption, want_tags)
                recovered = bool(caption or tags)
                if recovered:
                    logger.warning(
                        f"Salvaged partial JSON from {self.model_name} "
                        f"(caption={bool(caption)}, tags={len(tags)})."
                    )

        if not recovered:
            if looks_json:
                # Never let the free-text parser loose on JSON debris - it would emit
                # fragments like 'tags": ["dog' as a caption or a tag.
                raise VisionModelInvalidResponseError(
                    f"Malformed JSON response from {self.model_name}: {content[:200]}"
                )
            if structured is not None:
                logger.warning(f"Falling back to free-text parsing for {self.model_name} response.")
            else:
                logger.debug(f"Parsing free-text response from {self.model_name}")
            caption, tags = self._parse_freetext_response(content, want_caption, want_tags)

        # The response text was non-empty (guaranteed by _call_ollama), so recovering
        # nothing at all means the response was genuinely unusable.
        if not caption and not tags:
            raise VisionModelInvalidResponseError(
                f"Could not extract any metadata from {self.model_name} response: "
                f"{content[:200]}"
            )
        if want_caption and not caption:
            logger.warning(f"{self.model_name} returned tags but no caption.")
        if want_tags and not tags:
            logger.warning(f"{self.model_name} returned a caption but no tags.")
        return caption, tags

    def _generate_two_calls(
        self,
        *,
        image_base64: str,
        caption_instruction: str,
        tags_instruction: str,
        temperature: float,
        max_tokens: int,
        location_context: Optional[str],
        person_names: Optional[List[str]],
        total_faces: Optional[int],
        album_name: Optional[str],
    ) -> Tuple[str, List[str]]:
        """Legacy path: one request for the caption, one for the tags.

        Reachable via ``single_call=False``. Both requests receive the same context
        enrichment, so relationship/location context now reaches the tags too.

        Args:
            image_base64: Pre-encoded image (shared by both requests)
            caption_instruction: Caption instruction
            tags_instruction: Tags instruction
            temperature: Sampling temperature
            max_tokens: Maximum response tokens per request
            location_context: Optional resolved place name
            person_names: Optional recognized person names
            total_faces: Optional detected face count
            album_name: Optional album name

        Returns:
            Tuple of (caption, tags)
        """
        caption_prompt = self._build_combined_prompt(
            caption_instruction, "", structured=self.structured_output
        )
        caption_prompt = self._enhance_prompt_with_context(
            caption_prompt, location_context, person_names, total_faces, album_name
        )
        self._log_prompt("caption", caption_prompt)

        caption_schema = self._build_schema(True, False) if self.structured_output else None
        caption_content = self._call_ollama(
            caption_prompt, image_base64, temperature, max_tokens, response_format=caption_schema
        )
        caption, _ = self._interpret_response(
            caption_content, True, False, structured=caption_schema
        )

        tags_prompt = self._build_combined_prompt(
            "", tags_instruction, structured=self.structured_output
        )
        # Defect fix: the tags request gets the same context as the caption request.
        tags_prompt = self._enhance_prompt_with_context(
            tags_prompt, location_context, person_names, total_faces, album_name
        )
        self._log_prompt("tags", tags_prompt)

        tags_schema = self._build_schema(False, True) if self.structured_output else None
        tags_content = self._call_ollama(
            tags_prompt, image_base64, temperature, max_tokens, response_format=tags_schema
        )
        _, tags = self._interpret_response(tags_content, False, True, structured=tags_schema)

        return caption, tags

    def _log_prompt(self, label: str, prompt: str) -> None:
        """Log a full prompt at DEBUG level.

        Args:
            label: Short description of which prompt this is
            prompt: The prompt text
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(f"{label} prompt sent to {self.model_name}:")
        logger.debug("=" * 70)
        logger.debug(prompt)
        logger.debug("=" * 70)

    def generate_caption(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        location_context: Optional[str] = None,
        person_names: Optional[List[str]] = None,
        total_faces: Optional[int] = None,
    ) -> str:
        """Generate a caption for an image.

        Backward-compatible wrapper that delegates to :meth:`generate_metadata` with an
        empty tags instruction, so only the caption is requested.

        Args:
            image_path: Path to the image file
            prompt: Prompt text to guide caption generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens in response
            location_context: Optional location information to include in prompt
            person_names: Optional list of identified person names to include in prompt
            total_faces: Optional total number of faces detected (including unrecognized)

        Returns:
            Generated caption text

        Raises:
            VisionModelError: If caption generation fails
        """
        logger.debug(f"Generating caption for image: {image_path}")
        result = self.generate_metadata(
            image_path,
            prompt,
            "",
            temperature=temperature,
            max_tokens=max_tokens,
            location_context=location_context,
            person_names=person_names,
            total_faces=total_faces,
        )
        logger.debug(f"Generated caption: {result.caption[:100]}")
        return result.caption

    def generate_tags(
        self, image_path: str, prompt: str, temperature: float = 0.7, max_tokens: int = 150
    ) -> List[str]:
        """Generate keyword tags for an image.

        Backward-compatible wrapper that delegates to :meth:`generate_metadata` with an
        empty caption instruction, so only tags are requested.

        This signature carries no context arguments, so use :meth:`generate_metadata`
        directly when location/people/relationship context should influence the tags.

        Args:
            image_path: Path to the image file
            prompt: Prompt text to guide tag generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens in response

        Returns:
            List of generated keyword tags

        Raises:
            VisionModelError: If tag generation fails
        """
        logger.debug(f"Generating tags for image: {image_path}")
        result = self.generate_metadata(
            image_path, "", prompt, temperature=temperature, max_tokens=max_tokens
        )
        logger.debug(f"Generated {len(result.tags)} tags: {result.tags}")
        return result.tags

    def _parse_tags(self, response: str) -> List[str]:
        """Parse tags from a free-text model response - the legacy path.

        Still reachable with ``structured_output=False`` and as the fallback when a
        structured response cannot be parsed. The model may return tags in various
        formats:

        - "tag1, tag2, tag3"
        - "tag1,tag2,tag3"
        - "Tags: tag1, tag2, tag3"
        - Bullet points or numbered lists
        - Narrative text (extract keywords)

        Args:
            response: Raw response text from model

        Returns:
            List of cleaned, simple tag strings (preferably single words or short phrases)
        """
        # Remove common prefixes
        response = response.strip()
        prefixes = ["tags:", "keywords:", "tag list:", "tags are:", "the tags are:"]
        for prefix in prefixes:
            if response.lower().startswith(prefix):
                cut = len(prefix)
                response = response[cut:].strip()

        # First, try to find comma-separated lists
        # Look for patterns like "word1, word2, word3" or "word1,word2,word3"
        comma_pattern = r"([a-zA-Z][a-zA-Z\s-]{0,20})(?:,\s*|$)"
        comma_matches = re.findall(comma_pattern, response)

        if len(comma_matches) >= 3:  # If we found a good comma-separated list
            tags = [match.strip() for match in comma_matches]
        else:
            # Try splitting by common delimiters
            tags = []
            for delimiter in [",", ";", "\n", ". "]:
                if delimiter in response:
                    parts = response.split(delimiter)
                    if len(parts) >= 3:  # Looks like a list
                        tags = [part.strip() for part in parts]
                        break

            # If still no good list, try to extract keywords from narrative
            if not tags or len(tags) < 3:
                # Extract meaningful words (nouns, adjectives) from the text
                # Remove common stop words and extract capitalized or meaningful terms
                words = re.findall(r"\b([A-Z][a-z]+|[a-z]{4,})\b", response)
                # Filter out common words
                stop_words = {
                    "this",
                    "that",
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "from",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "being",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "could",
                    "should",
                    "may",
                    "might",
                    "can",
                    "image",
                    "features",
                    "showing",
                    "shows",
                    "visible",
                    "appears",
                    "characterized",
                }
                tags = [w for w in words if w.lower() not in stop_words and len(w) > 3]
                # Limit to most relevant (first 10-15)
                tags = tags[:15]

        # Clean up tags
        cleaned_tags = []
        for tag in tags:
            # Remove leading/trailing punctuation and whitespace
            tag = tag.strip(".,;:!?-()[]{}'\"")

            # Remove bullet points and numbering
            tag = re.sub(r"^[-*•]\s*", "", tag)  # Remove bullet
            tag = re.sub(r"^\d+\.\s*", "", tag)  # Remove numbering

            # Skip if too long (likely a sentence, not a tag)
            if len(tag) > 30:
                # Try to extract key words from long phrases
                words = tag.split()
                if len(words) > 3:
                    # Take first few meaningful words
                    tag = " ".join(words[:3])
                else:
                    continue

            # Skip empty tags or very short ones
            if tag and len(tag) >= 2:
                # Normalize: lowercase, remove extra spaces
                tag = " ".join(tag.split()).lower()

                # Filter out phrases that don't look like tags
                # Skip if it's a full sentence or contains common non-tag phrases
                skip_phrases = [
                    "do not",
                    "does not",
                    "is not",
                    "are not",
                    "was not",
                    "were not",
                    "seem",
                    "appears",
                    "looks like",
                    "appears to be",
                    "seems to",
                    "this image",
                    "the image",
                    "in the",
                    "on the",
                    "at the",
                    "characterized by",
                    "features a",
                    "showing",
                    "shows",
                    "the overall",
                    "the background",
                    "the foreground",
                ]

                # Skip if tag contains any of these phrases
                if any(phrase in tag for phrase in skip_phrases):
                    continue

                # Skip if tag is too long (likely a sentence fragment)
                if len(tag) > 25:
                    continue

                # Skip if tag contains too many words (likely a phrase, not a tag)
                if len(tag.split()) > 3:
                    continue

                cleaned_tags.append(tag)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in cleaned_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        # Limit to reasonable number of tags
        return unique_tags[:15]

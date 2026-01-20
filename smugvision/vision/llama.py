"""Llama 3.2 Vision model implementation via Ollama."""

import base64
import logging
import time
from pathlib import Path
from typing import List, Optional

import ollama
from PIL import Image

# Register HEIF/HEIC support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

from smugvision.vision.base import VisionModel, MetadataResult
from smugvision.vision.exceptions import (
    VisionModelError,
    VisionModelConnectionError,
    VisionModelTimeoutError,
    VisionModelInvalidResponseError,
    VisionModelImageError,
)

logger = logging.getLogger(__name__)


class LlamaVisionModel(VisionModel):
    """Llama 3.2 Vision model implementation using Ollama.
    
    This class provides integration with the Llama 3.2 Vision model
    running locally via Ollama. It handles image encoding, prompt
    formatting, and response parsing.
    
    Attributes:
        model_name: Name of the Ollama model (default: llama3.2-vision)
        endpoint: Ollama API endpoint URL
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2-vision",
        endpoint: Optional[str] = None,
        timeout: int = 120
    ) -> None:
        """Initialize Llama Vision model.
        
        Args:
            model_name: Name of the Ollama model
            endpoint: Optional Ollama endpoint URL (default: localhost:11434)
            timeout: Request timeout in seconds
            
        Raises:
            VisionModelConnectionError: If unable to connect to Ollama
        """
        super().__init__(model_name, endpoint)
        self.timeout = timeout
        
        # Configure Ollama client
        if endpoint:
            # Ollama client doesn't directly support custom endpoints in the
            # same way, but we can set the base URL via environment or client config
            # For now, we'll use the default client and log the endpoint
            logger.info(f"Using Ollama endpoint: {endpoint}")
        
        # Verify connection to Ollama
        try:
            self._verify_connection()
        except Exception as e:
            raise VisionModelConnectionError(
                f"Failed to connect to Ollama service: {e}"
            ) from e
    
    def _verify_connection(self) -> None:
        """Verify connection to Ollama service.
        
        Raises:
            VisionModelConnectionError: If connection fails
        """
        try:
            # Try to list models to verify connection
            # ollama.list() returns a dict with 'models' key containing list of model dicts
            models_response = ollama.list()
            
            # Handle different possible response structures
            if isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            elif isinstance(models_response, list):
                models_list = models_response
            else:
                models_list = []
            
            # Extract model names
            model_names = []
            for model in models_list:
                if isinstance(model, dict):
                    # Model dict can have 'name' or 'model' key
                    model_name = model.get('name') or model.get('model', '')
                    if model_name:
                        model_names.append(model_name)
                elif isinstance(model, str):
                    model_names.append(model)
            
            logger.debug(f"Connected to Ollama. Available models: {model_names}")
            
            # Check if our model is available
            # Note: Ollama might return empty list but model still works,
            # or model name might have slight variations
            if model_names and self.model_name not in model_names:
                # Check for partial matches (e.g., "llama3.2-vision" vs "llama3.2:vision")
                model_found = False
                for available_model in model_names:
                    if self.model_name in available_model or available_model in self.model_name:
                        model_found = True
                        logger.debug(
                            f"Model '{self.model_name}' matches available model: {available_model}"
                        )
                        break
                
                if not model_found:
                    # Only warn if we have a non-empty model list
                    logger.warning(
                        f"Model '{self.model_name}' not found in Ollama model list. "
                        f"Available models: {model_names}. "
                        f"The model may still work if it's available. "
                        f"If not, run: ollama pull {self.model_name}"
                    )
            elif not model_names:
                # Empty model list - model might still work, just log at debug level
                logger.debug(
                    f"Ollama model list is empty, but model '{self.model_name}' "
                    f"may still be available and work."
                )
        except Exception as e:
            raise VisionModelConnectionError(
                f"Cannot connect to Ollama service. "
                f"Make sure Ollama is running: {e}"
            ) from e
    
    def _strip_thinking_tags(self, content: str) -> str:
        """Strip <think>...</think> tags and other reasoning blocks from content.
        
        Some models (like MiniCPM-V 4.5) include their reasoning process in the
        response wrapped in <think> tags. This method removes those blocks and
        returns only the actual response.
        
        Args:
            content: Raw content that may contain thinking tags
            
        Returns:
            Cleaned content with thinking blocks removed
        """
        import re
        
        if not content:
            return content
        
        # Remove <think>...</think> blocks (including multiline)
        # Use DOTALL flag so . matches newlines
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Also handle unclosed <think> tags (model may have been cut off)
        # Remove everything from <think> to end if no closing tag
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up any leftover whitespace
        cleaned = cleaned.strip()
        
        # If we removed everything, the actual content might be after the think block
        # or the model only produced thinking - log this
        if content and not cleaned:
            logger.warning(
                "Content was entirely within <think> tags. "
                "Model may need prompt adjustment to disable thinking mode."
            )
        
        return cleaned
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string.
        
        Supports various image formats including JPEG, PNG, and HEIC/HEIF.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded image string
            
        Raises:
            VisionModelImageError: If image cannot be read or encoded
        """
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                raise VisionModelImageError(f"Image file not found: {image_path}")
            
            # Check for HEIC format
            file_ext = image_file.suffix.lower()
            is_heic = file_ext in ['.heic', '.heif']
            
            if is_heic and not HEIC_SUPPORT:
                raise VisionModelImageError(
                    f"HEIC format not supported. Install pillow-heif: "
                    f"pip install pillow-heif"
                )
            
            # Validate and open image
            try:
                # First verify the image is valid
                with Image.open(image_path) as img:
                    img.verify()
                
                # Reopen for actual reading (verify() closes the image)
                img = Image.open(image_path)
                try:
                    # Convert HEIC to RGB if needed (HEIC may be in different color space)
                    if is_heic and img.mode not in ('RGB', 'RGBA'):
                        logger.debug(f"Converting HEIC image from {img.mode} to RGB")
                        if img.mode == 'RGBA':
                            # Create white background for RGBA
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                            img.close()
                            img = background
                        else:
                            converted = img.convert('RGB')
                            img.close()
                            img = converted
                    
                    # Save to bytes in JPEG format for encoding
                    # This ensures compatibility and reduces size
                    from io import BytesIO
                    output = BytesIO()
                    img.save(output, format='JPEG', quality=85)  # Reduced from 95 to save memory
                    image_data = output.getvalue()
                    output.close()
                finally:
                    img.close()
                    
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
                f"Encoded image {image_path} ({len(image_data)} bytes, "
                f"format: {file_ext})"
            )
            return encoded
            
        except VisionModelImageError:
            raise
        except Exception as e:
            raise VisionModelImageError(
                f"Failed to encode image {image_path}: {e}"
            ) from e
    
    def _call_ollama(
        self,
        prompt: str,
        image_base64: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> str:
        """Call Ollama API with image and prompt.
        
        Args:
            prompt: Text prompt for the model
            image_base64: Base64-encoded image
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Model response text
            
        Raises:
            VisionModelTimeoutError: If request times out
            VisionModelInvalidResponseError: If response is invalid
            VisionModelError: For other errors
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ]
            
            logger.debug(
                f"Sending request to {self.model_name} "
                f"(temperature={temperature}, max_tokens={max_tokens})"
            )
            
            start_time = time.time()
            
            # Call Ollama with timeout handling
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"Received response in {elapsed:.2f}s")
            
            # Extract response text
            if not response or "message" not in response:
                raise VisionModelInvalidResponseError(
                    f"Invalid response structure from Ollama: {response}"
                )
            
            message = response.get("message", {})
            
            # Handle response - check both content and thinking fields
            # Some models (like Qwen3-VL) are "thinking" models that may put
            # their response in the thinking field instead of content
            content = ""
            
            # First try the standard content field
            if hasattr(message, "content"):
                content = message.content if message.content else ""
            elif isinstance(message, dict) and "content" in message:
                content = message["content"] or ""
            
            content = content.strip()
            
            # If content is empty, check for thinking field (for thinking models)
            if not content:
                thinking = None
                if hasattr(message, "thinking"):
                    thinking = message.thinking
                elif isinstance(message, dict) and "thinking" in message:
                    thinking = message.get("thinking")
                
                if thinking and isinstance(thinking, str):
                    # Extract useful content from thinking - look for the actual description
                    # Thinking often contains reasoning followed by the actual answer
                    thinking = thinking.strip()
                    logger.debug(f"Using thinking field content (model may be in thinking mode)")
                    
                    # Try to find a quoted description or the last complete sentence
                    # that looks like an actual caption
                    import re
                    
                    # Look for content in quotes (often the final answer)
                    quoted = re.findall(r'"([^"]{20,})"', thinking)
                    if quoted:
                        content = quoted[-1]  # Use the last quoted string
                    else:
                        # Use the thinking content directly, but clean it up
                        # Remove meta-commentary like "Got it, let's see" etc.
                        lines = thinking.split('\n')
                        useful_lines = []
                        for line in lines:
                            line = line.strip()
                            # Skip meta-commentary
                            if any(skip in line.lower() for skip in [
                                "let's see", "got it", "need to", "make sure",
                                "check", "first,", "so,", "identify"
                            ]):
                                continue
                            if line and len(line) > 20:
                                useful_lines.append(line)
                        
                        if useful_lines:
                            content = useful_lines[-1]  # Use last substantial line
                        else:
                            content = thinking[:500]  # Fallback to truncated thinking
            
            # Clean up thinking tags from models that include them in content
            # (e.g., MiniCPM-V 4.5 may include <think>...</think> blocks)
            content = self._strip_thinking_tags(content)
            
            if not content:
                raise VisionModelInvalidResponseError(
                    "Empty response from model (no content or thinking)"
                )
            
            return content
            
        except TimeoutError as e:
            raise VisionModelTimeoutError(
                f"Request to {self.model_name} timed out after {self.timeout}s"
            ) from e
        except Exception as e:
            if "timeout" in str(e).lower():
                raise VisionModelTimeoutError(
                    f"Request to {self.model_name} timed out: {e}"
                ) from e
            raise VisionModelError(
                f"Error calling Ollama model {self.model_name}: {e}"
            ) from e
    
    def generate_caption(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        location_context: Optional[str] = None,
        person_names: Optional[List[str]] = None,
        total_faces: Optional[int] = None
    ) -> str:
        """Generate a caption for an image.
        
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
        logger.info(f"Generating caption for image: {image_path}")
        
        try:
            # Enhance prompt with location and person context if available
            enhanced_prompt = self._enhance_prompt_with_context(
                prompt, location_context, person_names, total_faces
            )
            
            # Log the full prompt being sent to the model
            logger.info(f"Caption prompt sent to {self.model_name}:")
            logger.info(f"{'=' * 70}")
            logger.info(enhanced_prompt)
            logger.info(f"{'=' * 70}")
            
            # Encode image
            image_base64 = self._encode_image(image_path)
            
            # Call model
            caption = self._call_ollama(
                enhanced_prompt, image_base64, temperature, max_tokens
            )
            
            logger.debug(f"Generated caption: {caption[:100]}...")
            return caption
            
        except VisionModelError:
            raise
        except Exception as e:
            raise VisionModelError(
                f"Failed to generate caption for {image_path}: {e}"
            ) from e
    
    def _enhance_prompt_with_context(
        self,
        prompt: str,
        location_context: Optional[str],
        person_names: Optional[List[str]] = None,
        total_faces: Optional[int] = None
    ) -> str:
        """Enhance prompt with location and person context if available.
        
        Args:
            prompt: Original prompt text
            location_context: Optional location information
            person_names: Optional list of identified person names
            
        Returns:
            Enhanced prompt with context naturally incorporated
        """
        context_parts = []
        
        # Add person names with relationship context if available
        if person_names:
            # Try to load relationship context
            try:
                from smugvision.utils.relationships import get_relationship_manager
                rel_manager = get_relationship_manager()
                relationship_context = rel_manager.generate_context(person_names)
            except Exception as e:
                logger.debug(f"Could not load relationship context: {e}")
                relationship_context = None
            # Format names: replace underscores with spaces and capitalize properly
            formatted_names = [name.replace('_', ' ') for name in person_names]
            recognized_count = len(formatted_names)
            
            # Build person context - use relationship description if available
            if relationship_context:
                # We have relationship context, include both names and relationships
                names_list = ", ".join(formatted_names[:-1]) + f" and {formatted_names[-1]}" if len(formatted_names) > 1 else formatted_names[0]
                context_parts.append(
                    f"The people in this image are {names_list} ({relationship_context}). "
                    f"Please use their names and incorporate the relationship information naturally into your description."
                )
            else:
                # No relationship context, use names only
                # Handle case where there are more faces than recognized people
                if total_faces and total_faces > recognized_count:
                    # There are other people in the photo we couldn't identify
                    if recognized_count == 1:
                        context_parts.append(
                            f"There are {total_faces} people in this image. "
                            f"One of them is {formatted_names[0]}. "
                            f"Please use their name ({formatted_names[0]}) when describing them in the caption, "
                            f"and mention that there are other people present."
                        )
                    elif recognized_count == 2:
                        names_str = f"{formatted_names[0]} and {formatted_names[1]}"
                        context_parts.append(
                            f"There are {total_faces} people in this image. "
                            f"Two of them are {names_str}. "
                            f"Please use their names when describing them in the caption, "
                            f"and mention that there are other people present."
                        )
                    else:
                        names_str = ", ".join(formatted_names[:-1]) + f", and {formatted_names[-1]}"
                        context_parts.append(
                            f"There are {total_faces} people in this image. "
                            f"Some of them are {names_str}. "
                            f"Please use their names when describing them in the caption, "
                            f"and mention that there are other people present."
                        )
                else:
                    # All faces were recognized (or total_faces not provided)
                    if recognized_count == 1:
                        context_parts.append(
                            f"The person in this image is {formatted_names[0]}. "
                            f"Please use their name ({formatted_names[0]}) when describing them in the caption."
                        )
                    elif recognized_count == 2:
                        # Special formatting for two people: "Name1 and Name2"
                        names_str = f"{formatted_names[0]} and {formatted_names[1]}"
                        context_parts.append(
                            f"The people in this image are {names_str}. "
                            f"Please use their names when describing them in the caption."
                        )
                    else:
                        # Three or more: "Name1, Name2, and Name3"
                        names_str = ", ".join(formatted_names[:-1]) + f", and {formatted_names[-1]}"
                        context_parts.append(
                            f"The people in this image are {names_str}. "
                            f"Please use their names when describing them in the caption."
                        )
        
        # Add location context if available
        if location_context:
            context_parts.append(
                f"This image was taken at {location_context}."
            )
        
        if not context_parts:
            return prompt
        
        # Combine all context
        context_text = " ".join(context_parts)
        
        # Check if prompt already mentions location or people
        has_context_mention = any(
            keyword.lower() in prompt.lower()
            for keyword in ["location", "where", "place", "taken", "exif", "person", "people", "who"]
        )
        
        if has_context_mention:
            # If prompt already mentions context, append it
            enhanced = f"{prompt}\n\nAdditional context: {context_text} Please incorporate this information naturally into the caption."
        else:
            # Otherwise, add it as context with explicit instruction
            enhanced = f"{prompt}\n\nContext: {context_text} Please incorporate this information naturally into your description, including using the person's name when referring to them."
        
        logger.debug(
            f"Enhanced prompt with context: location={location_context}, "
            f"people={person_names}"
        )
        return enhanced
    
    def generate_tags(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> List[str]:
        """Generate keyword tags for an image.
        
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
        logger.info(f"Generating tags for image: {image_path}")
        
        try:
            # Log the full prompt being sent to the model
            logger.info(f"Tags prompt sent to {self.model_name}:")
            logger.info(f"{'=' * 70}")
            logger.info(prompt)
            logger.info(f"{'=' * 70}")
            
            # Encode image
            image_base64 = self._encode_image(image_path)
            
            # Call model
            response = self._call_ollama(prompt, image_base64, temperature, max_tokens)
            
            # Parse tags from response
            tags = self._parse_tags(response)
            
            logger.debug(f"Generated {len(tags)} tags: {tags}")
            return tags
            
        except VisionModelError:
            raise
        except Exception as e:
            raise VisionModelError(
                f"Failed to generate tags for {image_path}: {e}"
            ) from e
    
    def _parse_tags(self, response: str) -> List[str]:
        """Parse tags from model response, extracting simple keywords.
        
        The model may return tags in various formats:
        - "tag1, tag2, tag3"
        - "tag1,tag2,tag3"
        - "Tags: tag1, tag2, tag3"
        - Bullet points or numbered lists
        - Narrative text (extract keywords)
        
        This method attempts to extract simple, short tags from various formats.
        
        Args:
            response: Raw response text from model
            
        Returns:
            List of cleaned, simple tag strings (preferably single words or short phrases)
        """
        import re
        
        # Remove common prefixes
        response = response.strip()
        prefixes = ["tags:", "keywords:", "tag list:", "tags are:", "the tags are:"]
        for prefix in prefixes:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        
        # First, try to find comma-separated lists
        # Look for patterns like "word1, word2, word3" or "word1,word2,word3"
        comma_pattern = r'([a-zA-Z][a-zA-Z\s-]{0,20})(?:,\s*|$)'
        comma_matches = re.findall(comma_pattern, response)
        
        if len(comma_matches) >= 3:  # If we found a good comma-separated list
            tags = [match.strip() for match in comma_matches]
        else:
            # Try splitting by common delimiters
            tags = []
            for delimiter in [',', ';', '\n', '. ']:
                if delimiter in response:
                    parts = response.split(delimiter)
                    if len(parts) >= 3:  # Looks like a list
                        tags = [part.strip() for part in parts]
                        break
            
            # If still no good list, try to extract keywords from narrative
            if not tags or len(tags) < 3:
                # Extract meaningful words (nouns, adjectives) from the text
                # Remove common stop words and extract capitalized or meaningful terms
                words = re.findall(r'\b([A-Z][a-z]+|[a-z]{4,})\b', response)
                # Filter out common words
                stop_words = {
                    'this', 'that', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                    'at', 'to', 'for', 'of', 'with', 'from', 'is', 'are', 'was', 'were',
                    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'image',
                    'features', 'showing', 'shows', 'visible', 'appears', 'characterized'
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
            tag = re.sub(r'^[-*•]\s*', '', tag)  # Remove bullet
            tag = re.sub(r'^\d+\.\s*', '', tag)  # Remove numbering
            
            # Skip if too long (likely a sentence, not a tag)
            if len(tag) > 30:
                # Try to extract key words from long phrases
                words = tag.split()
                if len(words) > 3:
                    # Take first few meaningful words
                    tag = ' '.join(words[:3])
                else:
                    continue
            
            # Skip empty tags or very short ones
            if tag and len(tag) >= 2:
                # Normalize: lowercase, remove extra spaces
                tag = ' '.join(tag.split()).lower()
                
                # Filter out phrases that don't look like tags
                # Skip if it's a full sentence or contains common non-tag phrases
                skip_phrases = [
                    'do not', 'does not', 'is not', 'are not', 'was not', 'were not',
                    'seem', 'appears', 'looks like', 'appears to be', 'seems to',
                    'this image', 'the image', 'in the', 'on the', 'at the',
                    'characterized by', 'features a', 'showing', 'shows',
                    'the overall', 'the background', 'the foreground'
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


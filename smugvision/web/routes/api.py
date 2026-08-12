"""REST API routes for smugVision web UI."""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict
from flask import Blueprint, request, jsonify, current_app, Response, send_file

from ...smugmug import (
    SmugMugAPIError,
    SmugMugAuthError,
    SmugMugError,
    SmugMugNotFoundError,
    SmugMugRateLimitError,
)
from ..services.preview import (
    ConfirmationRequiredError,
    ImageNotInJobError,
    JobNotFoundError,
    JobNotReadyError,
    PreviewService,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

# Module-level singleton for preview service (thread-safe)
_preview_service: PreviewService = None
_preview_service_lock = threading.Lock()


def get_preview_service() -> PreviewService:
    """Get or create preview service singleton.
    
    Uses a module-level singleton with thread locking to ensure
    the service is only created once across all requests.
    """
    global _preview_service
    
    if _preview_service is None:
        with _preview_service_lock:
            # Double-check after acquiring lock
            if _preview_service is None:
                config = current_app.config["SMUGVISION_CONFIG"]
                _preview_service = PreviewService(config)
                logger.info("Created PreviewService singleton")
    
    return _preview_service


def _json_body() -> Dict[str, Any]:
    """Read the request body as a JSON object.

    A missing, unparseable or non-object body reads as an empty dict so each route can
    report the field it actually needs instead of a generic parse error.

    Returns:
        The decoded JSON object, or an empty dict
    """
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _string_field(data: Dict[str, Any], name: str) -> str:
    """Read one field of a JSON body as a trimmed string.

    Args:
        data: Decoded JSON object
        name: Field name to read

    Returns:
        The trimmed string value; "" when the field is absent or not textual
    """
    value = data.get(name)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value).strip()
    return ""


def _is_truthy(value: str) -> bool:
    """Interpret a query-string flag.

    Args:
        value: Raw query parameter value

    Returns:
        True for "1", "true", "yes" and "on" (case-insensitive)
    """
    return value.strip().lower() in ("1", "true", "yes", "on")


@api_bp.route("/galleries", methods=["GET"])
def list_galleries():
    """List one level of the SmugMug folder tree for the gallery picker.

    Query params:
        node: Node ID to list. Omit to start at the user's root node.
        refresh: Set to 1/true to bypass the short-lived listing cache.

    Returns:
        JSON with the node, its breadcrumb (root first), child folders, child albums,
        totals, and whether the response came from cache
    """
    node_id = request.args.get("node") or None
    refresh = _is_truthy(request.args.get("refresh", ""))

    try:
        service = get_preview_service()
        return jsonify(service.list_galleries(node_id=node_id, refresh=refresh))

    except SmugMugNotFoundError:
        return jsonify({"error": f"Node not found: {node_id or '<root>'}"}), 404
    except SmugMugAuthError as e:
        return jsonify({"error": f"SmugMug authentication failed: {e}"}), 401
    except SmugMugRateLimitError as e:
        return jsonify({"error": f"SmugMug rate limit reached: {e}"}), 429
    except SmugMugAPIError as e:
        message = str(e)
        # list_node_children raises this deliberately when asked to browse INTO an
        # album; that is a bad request, not an upstream failure.
        status = 400 if "not a folder" in message else 502
        return jsonify({"error": message}), status
    except Exception as e:
        logger.error(f"Failed to list galleries for node {node_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/preview", methods=["POST"])
def start_preview():
    """Start a preview processing job for an album.

    The album is identified either by URL (paste-a-link) or by album key (gallery
    picker). Preview is always dry-run: nothing is written to SmugMug until
    ``POST /api/commit`` runs with explicit confirmation.

    Request body:
        {
            "url": "https://site.smugmug.com/.../n-XXXXX/album-name",
            "album_key": "Ab3kZq",
            "force_reprocess": false
        }

    Exactly one of "url" or "album_key" is required.

    Returns:
        Job information including job_id for tracking
    """
    data = _json_body()

    url = _string_field(data, "url")
    album_key = _string_field(data, "album_key")

    if not url and not album_key:
        return jsonify({"error": "Missing 'url' or 'album_key' in request body"}), 400
    if url and album_key:
        return jsonify({"error": "Provide either 'url' or 'album_key', not both"}), 400

    force_reprocess = bool(data.get("force_reprocess", False))

    try:
        service = get_preview_service()
        job = service.create_preview_job(
            url=url or None,
            force_reprocess=force_reprocess,
            album_key=album_key or None,
        )

        return jsonify({
            "job_id": job.job_id,
            "album_key": job.album_key,
            "album_name": job.album_name,
            "total_images": job.total_images,
            "status": job.status,
        })

    except ValueError as e:
        logger.error(f"Invalid preview request: {e}")
        return jsonify({"error": str(e)}), 400
    except SmugMugNotFoundError as e:
        logger.error(f"Album not found: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to create preview job: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/preview/status", methods=["GET"])
def preview_status():
    """Stream preview progress via Server-Sent Events.
    
    Query params:
        job_id: Preview job ID
        force_reprocess: Whether to reprocess tagged images (optional)
        
    Returns:
        SSE stream with progress events
    """
    job_id = request.args.get("job_id")
    force_reprocess = request.args.get("force_reprocess", "false").lower() == "true"
    
    if not job_id:
        return jsonify({"error": "Missing 'job_id' query parameter"}), 400
    
    service = get_preview_service()
    job = service.get_job(job_id)
    
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    
    def generate():
        """Generate SSE events."""
        try:
            for event in service.process_preview(job_id, force_reprocess):
                event_type = event.get("event", "message")
                event_data = json.dumps(event.get("data", {}))
                logger.debug(f"SSE sending event: {event_type}")
                # SSE format: event type, data, then double newline
                yield f"event: {event_type}\ndata: {event_data}\n\n"
            
            logger.info("SSE stream completed - all events sent")
        except GeneratorExit:
            # Client disconnected
            logger.info("SSE client disconnected")
        except Exception as e:
            logger.error(f"SSE stream error: {e}", exc_info=True)
            error_data = json.dumps({"message": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"
    
    response = Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
    response.implicit_sequence_conversion = False
    return response


@api_bp.route("/preview/results", methods=["GET"])
def preview_results():
    """Get full preview results for a completed job.
    
    Query params:
        job_id: Preview job ID
        
    Returns:
        Full job results including all image preview data
    """
    job_id = request.args.get("job_id")
    
    if not job_id:
        return jsonify({"error": "Missing 'job_id' query parameter"}), 400
    
    service = get_preview_service()
    job = service.get_job(job_id)
    
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    
    return jsonify({
        "job_id": job.job_id,
        "album_key": job.album_key,
        "album_name": job.album_name,
        "status": job.status,
        "stats": {
            "total": job.total_images,
            "processed": job.processed_count,
            "skipped": job.skipped_count,
            "errors": job.error_count,
        },
        "images": [result.to_dict() for result in job.results],
        "hints": _job_hints(service, job),
        "error": job.error,
    })


def _job_hints(service: PreviewService, job) -> Dict[str, Any]:
    """Collect the hints that apply to one job, so a page needs a single fetch.

    Args:
        service: Preview service holding the hint manager
        job: PreviewJob whose album and images should be reported

    Returns:
        ``{"enabled": bool, "global": str, "album": str, "images": {image_key: str}}``
    """
    manager = service.hints
    if manager is None:
        return {"enabled": False, "global": "", "album": "", "images": {}}

    try:
        stored = manager.get_all()
    except Exception as e:
        logger.warning(f"Could not read hints for job {job.job_id}: {e}")
        return {"enabled": False, "global": "", "album": "", "images": {}}

    images = stored.get("images", {})
    return {
        "enabled": True,
        "global": stored.get("global", ""),
        "album": stored.get("albums", {}).get(job.album_key, ""),
        "images": {
            result.image_key: images.get(result.image_key, "")
            for result in job.results
        },
    }


@api_bp.route("/preview/<job_id>/regenerate", methods=["POST"])
def regenerate_preview_image(job_id: str):
    """Re-run a single image of a finished job with the hints that apply right now.

    Still dry-run: this re-processes one image through the same dry-run
    ImageProcessor and replaces that one entry in the job. No other image is touched
    and nothing is written to SmugMug.

    Request body:
        {
            "image_key": "Xy7NpQr",
            "force_reprocess": true
        }

    Returns:
        The updated image result, the hint text that was applied, and the job's
        refreshed statistics. Status codes: 400 missing image_key, 404 unknown job,
        409 job still running, 422 image not in this job, 502 vision model or SmugMug
        failure.
    """
    data = _json_body()
    image_key = _string_field(data, "image_key")

    if not image_key:
        return jsonify({"error": "Missing 'image_key' in request body"}), 400

    force_reprocess = bool(data.get("force_reprocess", True))

    service = get_preview_service()

    try:
        result = service.regenerate_image(
            job_id=job_id,
            image_key=image_key,
            force_reprocess=force_reprocess,
        )
    except JobNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except JobNotReadyError as e:
        return jsonify({"error": str(e)}), 409
    except ImageNotInJobError as e:
        return jsonify({"error": str(e)}), 422
    except SmugMugError as e:
        logger.error(f"SmugMug error regenerating {image_key} in job {job_id}: {e}")
        return jsonify({"error": f"SmugMug request failed: {e}"}), 502
    except Exception as e:
        logger.error(
            f"Failed to regenerate {image_key} in job {job_id}: {e}", exc_info=True
        )
        return jsonify({"error": str(e)}), 500

    job = service.get_job(job_id)
    manager = service.hints
    hint_applied = ""
    if manager is not None and job is not None:
        try:
            hint_applied = manager.resolve(job.album_key, image_key)
        except Exception as e:
            logger.warning(f"Could not resolve hint text for {image_key}: {e}")

    payload: Dict[str, Any] = {
        "job_id": job_id,
        "image": result.to_dict(),
        "hint_applied": hint_applied,
        "stats": {
            "total": job.total_images if job else 0,
            "processed": job.processed_count if job else 0,
            "skipped": job.skipped_count if job else 0,
            "errors": job.error_count if job else 0,
        },
    }

    if result.status == "error":
        # The image itself is still returned so the card can show what failed.
        payload["error"] = result.error or "Regeneration failed"
        return jsonify(payload), 502

    return jsonify(payload)


@api_bp.route("/hints", methods=["GET"])
def get_hints():
    """Get every stored hint, by scope.

    Returns:
        ``{"global": str, "albums": {album_key: str}, "images": {image_key: str}}``
        plus whether hints are enabled, the file backing them, and a count
    """
    try:
        service = get_preview_service()
        manager = service.hints

        if manager is None:
            return jsonify({
                "global": "",
                "albums": {},
                "images": {},
                "enabled": False,
                "message": "Hints are disabled (hints.enabled=false in config)",
            })

        payload: Dict[str, Any] = dict(manager.get_all())
        payload["enabled"] = True
        payload["file"] = str(manager.hints_file)
        payload["count"] = manager.hint_count
        return jsonify(payload)

    except Exception as e:
        logger.error(f"Failed to read hints: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/hints", methods=["PUT"])
def put_hint():
    """Create, update or clear one hint.

    Hints are user-asserted facts about a photo that outrank the model's own visual
    guess. Scope is one of "global", "album" or "image"; "album" and "image" require
    "key" (the SmugMug album/image key) and "global" must not have one. Empty text
    clears the hint.

    Request body:
        {
            "scope": "image",
            "key": "Xy7NpQr",
            "text": "The white ribbed object is a Nylabone dog chew."
        }

    Returns:
        The scope, key and the value that is now stored (empty when cleared)
    """
    data = _json_body()

    scope = _string_field(data, "scope")
    key = _string_field(data, "key") or None
    text = data.get("text", "")

    if not scope:
        return jsonify({"error": "Missing 'scope' in request body"}), 400
    if not isinstance(text, str):
        return jsonify({"error": "'text' must be a string"}), 400

    try:
        service = get_preview_service()
        manager = service.hints

        if manager is None:
            return jsonify({
                "error": "Hints are disabled (hints.enabled=false in config); "
                         "nothing was saved"
            }), 409

        # HintManager owns scope validation, so the API cannot drift from the CLI.
        manager.set_hint(scope, text, key)

        normalized = scope.strip().lower()
        stored = manager.get_all()
        if normalized == "global":
            value = stored.get("global", "")
        elif normalized == "album":
            value = stored.get("albums", {}).get(key, "")
        else:
            value = stored.get("images", {}).get(key, "")

        return jsonify({
            "scope": normalized,
            "key": key,
            "text": value,
            "cleared": not value,
            "count": manager.hint_count,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        logger.error(f"Could not persist hint: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Failed to store hint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/commit", methods=["POST"])
def commit_changes():
    """Commit previewed changes to SmugMug. THIS WRITES TO THE USER'S ACCOUNT.

    Confirmation is mandatory: without ``"confirm": true`` the request is refused with
    HTTP 400 and nothing is sent to SmugMug.

    Request body:
        {
            "job_id": "abc123",
            "confirm": true,
            "image_keys": ["Xy7NpQr"]
        }

    "image_keys" is optional; omit it to commit every processed image in the job.

    Returns:
        Commit results. Status codes: 400 missing job_id / unconfirmed / bad
        image_keys, 404 unknown job, 409 job not complete.
    """
    data = _json_body()

    job_id = _string_field(data, "job_id")
    if not job_id:
        return jsonify({"error": "Missing 'job_id' in request body"}), 400

    confirm = data.get("confirm", False)
    image_keys = data.get("image_keys")

    if image_keys is not None and not isinstance(image_keys, list):
        return jsonify({"error": "'image_keys' must be a list of SmugMug image keys"}), 400

    try:
        service = get_preview_service()
        # confirm is passed straight through; the service is the one that refuses, so a
        # non-web caller cannot skip the check.
        result = service.commit_changes(
            job_id=job_id,
            confirm=confirm is True,
            image_keys=image_keys,
        )
        return jsonify(result)

    except ConfirmationRequiredError as e:
        logger.warning(f"Commit refused for job {job_id}: {e}")
        return jsonify({"error": str(e), "committed": 0}), 400
    except JobNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except JobNotReadyError as e:
        return jsonify({"error": str(e)}), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to commit changes: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/thumbnail/<image_key>", methods=["GET"])
def get_thumbnail(image_key: str):
    """Serve thumbnail from local cache.
    
    Images are already downloaded during preview processing, so we just
    serve them directly from the cache directory.
    
    Args:
        image_key: SmugMug image key
        
    Returns:
        Image binary
    """
    try:
        service = get_preview_service()
        
        # Find the cached image file from preview job results
        for job in service._jobs.values():
            for result in job.results:
                if result.image_key == image_key:
                    # Get the album name to find the cache directory
                    album_cache_dir = service.cache.get_album_cache_dir(
                        album_name=job.album_name,
                        folder_path=None
                    )
                    
                    # Build path to cached image
                    image_path = album_cache_dir / result.filename
                    
                    if image_path.exists():
                        # Determine mime type from extension
                        ext = image_path.suffix.lower()
                        mime_types = {
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.heic': 'image/heic',
                            '.heif': 'image/heif',
                            '.gif': 'image/gif',
                        }
                        mimetype = mime_types.get(ext, 'image/jpeg')
                        
                        return send_file(
                            image_path,
                            mimetype=mimetype,
                            max_age=3600  # Cache for 1 hour
                        )
                    else:
                        logger.warning(f"Cached image not found: {image_path}")
        
        # Image not found in any job
        logger.warning(f"No cached image found for key {image_key}")
        return jsonify({"error": "Image not found in cache"}), 404
        
    except Exception as e:
        logger.error(f"Failed to serve thumbnail for {image_key}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/faces", methods=["GET"])
def list_faces():
    """Get list of known reference faces.
    
    Returns:
        List of known people with face counts
    """
    try:
        service = get_preview_service()
        
        if not service.face_recognizer:
            return jsonify({
                "faces": [],
                "total": 0,
                "enabled": False,
                "message": "Face recognition is not enabled or configured"
            })
        
        faces = []
        for name, encodings in service.face_recognizer.reference_faces.items():
            faces.append({
                "name": name,
                "display_name": name.replace("_", " "),
                "reference_count": len(encodings),
            })
        
        # Sort by display name
        faces.sort(key=lambda f: f["display_name"])
        
        return jsonify({
            "faces": faces,
            "total": len(faces),
            "enabled": True,
        })
        
    except Exception as e:
        logger.error(f"Failed to list faces: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/face-sample/<person_name>", methods=["GET"])
def get_face_sample(person_name: str):
    """Get a sample reference face image for a person.
    
    Args:
        person_name: Person name (with underscores)
        
    Returns:
        Image binary
    """
    try:
        config = current_app.config["SMUGVISION_CONFIG"]
        reference_faces_dir = config.get(
            "face_recognition.reference_faces_dir",
            "~/.smugvision/reference_faces"
        )
        reference_faces_path = Path(reference_faces_dir).expanduser()
        
        person_dir = reference_faces_path / person_name
        if not person_dir.exists():
            return jsonify({"error": f"Person {person_name} not found"}), 404
        
        # Find first image in the person's directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        for image_path in person_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                # Return the image
                return send_file(
                    image_path,
                    mimetype=f"image/{image_path.suffix[1:].lower()}"
                )
        
        return jsonify({"error": "No sample image found"}), 404
        
    except Exception as e:
        logger.error(f"Failed to get face sample for {person_name}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/relationships", methods=["GET"])
def get_relationships():
    """Get relationship graph data.
    
    Returns:
        Nodes, edges, and groups for visualization
    """
    try:
        from ...utils.relationships import RelationshipManager
        
        config = current_app.config["SMUGVISION_CONFIG"]
        relationship_manager = RelationshipManager()
        
        if not relationship_manager.enabled:
            return jsonify({
                "nodes": [],
                "edges": [],
                "groups": [],
                "enabled": False,
                "message": "Relationships not configured"
            })
        
        # Build nodes from unique people in relationships
        people = set()
        for person1, person2, rel_type in relationship_manager.relationships:
            people.add(person1)
            people.add(person2)
        
        nodes = [
            {"id": name, "label": name.replace("_", " ")}
            for name in sorted(people)
        ]
        
        # Build edges from relationships
        edges = [
            {"from": person1, "to": person2, "label": rel_type}
            for person1, person2, rel_type in relationship_manager.relationships
        ]
        
        # Include groups
        groups = [
            {
                "members": group.get("members", []),
                "description": group.get("description", "")
            }
            for group in relationship_manager.groups
        ]
        
        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "groups": groups,
            "enabled": True,
        })
        
    except Exception as e:
        logger.error(f"Failed to get relationships: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/status", methods=["GET"])
def api_status():
    """Get API and service status.
    
    Returns:
        Status of various services (SmugMug, Ollama, face recognition)
    """
    try:
        config = current_app.config["SMUGVISION_CONFIG"]
        
        status = {
            "api": "ok",
            "config_loaded": True,
            "smugmug": "unknown",
            "vision_model": "unknown",
            "face_recognition": "unknown",
        }
        
        # Check SmugMug
        try:
            service = get_preview_service()
            # Just accessing smugmug property triggers auth check
            _ = service.smugmug
            status["smugmug"] = "connected"
        except Exception as e:
            status["smugmug"] = f"error: {str(e)}"
        
        # Check vision model
        try:
            model_name = config.get("vision.model", "llama3.2-vision")
            endpoint = config.get("vision.endpoint", "http://localhost:11434")
            status["vision_model"] = f"{model_name} at {endpoint}"
            
            # Try to ping Ollama
            import requests
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            if response.ok:
                status["vision_model"] += " (connected)"
            else:
                status["vision_model"] += " (not responding)"
        except Exception as e:
            status["vision_model"] = f"error: {str(e)}"
        
        # Check face recognition
        try:
            if config.get("face_recognition.enabled", True):
                service = get_preview_service()
                if service.face_recognizer:
                    count = len(service.face_recognizer.reference_faces)
                    status["face_recognition"] = f"enabled ({count} people)"
                else:
                    status["face_recognition"] = "enabled but not configured"
            else:
                status["face_recognition"] = "disabled"
        except Exception as e:
            status["face_recognition"] = f"error: {str(e)}"
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return jsonify({"api": "error", "error": str(e)}), 500

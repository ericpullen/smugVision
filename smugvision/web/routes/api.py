"""REST API routes for smugVision web UI."""

import json
import logging
import threading
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, Response, send_file

from ..services.preview import PreviewService

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


@api_bp.route("/preview", methods=["POST"])
def start_preview():
    """Start a preview processing job for an album.
    
    Request body:
        {
            "url": "https://site.smugmug.com/.../n-XXXXX/album-name",
            "force_reprocess": false
        }
        
    Returns:
        Job information including job_id for tracking
    """
    data = request.get_json()
    
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    
    url = data["url"]
    force_reprocess = data.get("force_reprocess", False)
    
    try:
        service = get_preview_service()
        job = service.create_preview_job(url, force_reprocess)
        
        return jsonify({
            "job_id": job.job_id,
            "album_key": job.album_key,
            "album_name": job.album_name,
            "total_images": job.total_images,
            "status": job.status,
        })
        
    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        return jsonify({"error": str(e)}), 400
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
        "error": job.error,
    })


@api_bp.route("/commit", methods=["POST"])
def commit_changes():
    """Commit previewed changes to SmugMug.
    
    Request body:
        {
            "job_id": "abc123"
        }
        
    Returns:
        Commit results
    """
    data = request.get_json()
    
    if not data or "job_id" not in data:
        return jsonify({"error": "Missing 'job_id' in request body"}), 400
    
    job_id = data["job_id"]
    
    try:
        service = get_preview_service()
        result = service.commit_changes(job_id)
        return jsonify(result)
        
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

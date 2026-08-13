"""HTML page routes for smugVision web UI."""

import logging
from flask import Blueprint, render_template, current_app
from jinja2 import TemplateNotFound

logger = logging.getLogger(__name__)

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Main page: gallery picker plus the paste-an-album-URL fallback."""
    return render_template("index.html")


@pages_bp.route("/hints")
def hints():
    """Hint editor page.

    The template is optional so this route can exist ahead of the page itself; a
    missing template reports a clear 404 instead of a server error.

    Returns:
        Rendered hint editor, or a 404 message when the template is not installed
    """
    try:
        return render_template("hints.html")
    except TemplateNotFound:
        logger.info("Hint editor page requested but templates/hints.html is not installed")
        return (
            "Hint editor page is not installed (templates/hints.html is missing). "
            "Hints are still available through the API at /api/hints.",
            404,
        )


@pages_bp.route("/preview/<job_id>")
def preview(job_id: str):
    """Preview results page for a specific job."""
    return render_template("preview.html", job_id=job_id)


@pages_bp.route("/faces")
def faces():
    """Known faces display page."""
    return render_template("faces.html")


@pages_bp.route("/relationships")
def relationships():
    """Relationship graph page."""
    return render_template("relationships.html")

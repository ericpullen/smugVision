"""HTML page routes for smugVision web UI."""

import logging
from flask import Blueprint, render_template, current_app

logger = logging.getLogger(__name__)

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Main page with album URL input."""
    return render_template("index.html")


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

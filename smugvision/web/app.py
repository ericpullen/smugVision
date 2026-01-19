"""Flask application factory for smugVision web UI."""

import logging
from pathlib import Path
from flask import Flask

from ..config import ConfigManager

logger = logging.getLogger(__name__)


def create_app(config_path: str = None, debug: bool = False) -> Flask:
    """Create and configure the Flask application.
    
    Args:
        config_path: Optional path to smugVision config file
        debug: Enable debug mode
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    
    app.config["DEBUG"] = debug
    app.config["SECRET_KEY"] = "smugvision-local-dev-key"  # Local only, no real security needed
    
    # Load smugVision config
    try:
        smugvision_config = ConfigManager.load(
            config_path=config_path,
            interactive=False
        )
        app.config["SMUGVISION_CONFIG"] = smugvision_config
        logger.info(f"Loaded smugVision config from: {smugvision_config.config_path}")
    except Exception as e:
        logger.error(f"Failed to load smugVision config: {e}")
        raise
    
    # Register blueprints
    from .routes.pages import pages_bp
    from .routes.api import api_bp
    
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    
    logger.info("smugVision web app created")
    
    return app

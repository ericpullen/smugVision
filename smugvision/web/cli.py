#!/usr/bin/env python3
"""CLI entry point for smugVision web UI server."""

import argparse
import logging
import sys

from .app import create_app


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the web server.
    
    Args:
        verbose: If True, enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Reduce noise from werkzeug in non-debug mode
    if not verbose:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="smugVision Web UI - Local web interface for photo metadata preview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start web server on default port
  smugvision-web
  
  # Start on custom port
  smugvision-web --port 8080
  
  # Enable debug mode
  smugvision-web --debug
  
  # Use custom config file
  smugvision-web --config /path/to/config.yaml

The web UI will be available at http://localhost:5050 (or your chosen port).
"""
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5050,
        help="Port to run the server on (default: 5050)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, localhost only)"
    )
    
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to smugVision config file (default: ~/.smugvision/config.yaml)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with auto-reload"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def print_banner(host: str, port: int) -> None:
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  smugVision Web UI")
    print("=" * 60)
    print()
    print(f"  Server running at: http://{host}:{port}")
    print()
    print("  Pages:")
    print(f"    - Process Album: http://{host}:{port}/")
    print(f"    - Known Faces:   http://{host}:{port}/faces")
    print(f"    - Relationships: http://{host}:{port}/relationships")
    print()
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()


def main() -> int:
    """Main entry point for smugVision web server.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose or args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Create Flask app
        app = create_app(
            config_path=args.config,
            debug=args.debug
        )
        
        # Print banner
        print_banner(args.host, args.port)
        
        # Run server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,  # Handle multiple requests
            use_reloader=args.debug  # Auto-reload in debug mode
        )
        
        return 0
        
    except KeyboardInterrupt:
        print()
        print("Server stopped.")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=args.verbose)
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

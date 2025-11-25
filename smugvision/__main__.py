#!/usr/bin/env python3
"""smugVision - AI-powered photo metadata generation for SmugMug.

This is the main CLI entry point for smugVision. It processes SmugMug albums
and automatically generates captions and tags using local AI vision models.

Usage:
    python -m smugvision --gallery <album_key>
    python -m smugvision --url "https://site.smugmug.com/.../album-name"
    python -m smugvision --gallery <album_key> --dry-run
    python -m smugvision --gallery <album_key> --force-reprocess
"""

import sys
import logging
import argparse
import re
from urllib.parse import urlparse
from pathlib import Path

from ._version import __version__
from .config import ConfigManager
from .processing import ImageProcessor
from .smugmug import SmugMugClient
from .smugmug.exceptions import SmugMugError
from .config.manager import ConfigError


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.
    
    Args:
        verbose: If True, enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler with simpler format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Also log to file if configured
    try:
        config = ConfigManager.load(interactive=False)
        log_file = config.get("logging.file")
        if log_file:
            log_path = Path(log_file).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)  # Always DEBUG in file
            file_formatter = logging.Formatter(
                config.get(
                    "logging.format",
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
    except Exception:
        # If logging setup fails, just continue with console logging
        pass


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="smugVision - AI-powered photo metadata generation for SmugMug",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process an album by key
  python -m smugvision --gallery abc123
  
  # Process an album by URL
  python -m smugvision --url "https://site.smugmug.com/.../n-XXXXX/album-name"
  
  # Preview without updating SmugMug
  python -m smugvision --gallery abc123 --dry-run
  
  # Force reprocess already-tagged images
  python -m smugvision --gallery abc123 --force-reprocess
  
  # Skip video files (default behavior)
  python -m smugvision --gallery abc123
  
  # Include video files
  python -m smugvision --gallery abc123 --include-videos

For more information, visit: https://github.com/yourusername/smugvision
"""
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"smugVision {__version__}"
    )
    
    # Album selection (mutually exclusive)
    album_group = parser.add_mutually_exclusive_group(required=True)
    album_group.add_argument(
        "--gallery",
        metavar="KEY",
        help="SmugMug album key to process"
    )
    album_group.add_argument(
        "--url",
        metavar="URL",
        help="SmugMug album URL (auto-extracts album key)"
    )
    
    # Processing options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview processing without updating SmugMug"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess images even if already tagged with marker"
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Include video files in processing (default: skip videos)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to config file (default: ~/.smugvision/config.yaml)"
    )
    
    # Output control
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    return parser.parse_args()


def resolve_album_key(args: argparse.Namespace, config: ConfigManager) -> tuple:
    """Resolve album identifier to album key.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        Tuple of (album_key, album_name) where album_name is for display
        
    Raises:
        SmugMugError: If album cannot be resolved
    """
    if args.gallery:
        return args.gallery, None
    
    # Parse URL to extract node ID and album name
    node_match = re.search(r'/n-([a-zA-Z0-9]+)', args.url)
    if not node_match:
        raise ValueError(
            "Could not extract node ID from URL. "
            "Expected format: .../n-XXXXX/album-name"
        )
    
    node_id = node_match.group(1)
    
    # Get album name from URL
    path = urlparse(args.url).path
    parts = [p for p in path.split('/') if p and not p.startswith('n-')]
    if not parts:
        raise ValueError(
            "Could not extract album name from URL. "
            "Expected format: .../n-XXXXX/album-name"
        )
    
    album_identifier = parts[-1]
    
    print(f"Resolving album from URL...")
    print(f"  Node ID: {node_id}")
    print(f"  Album identifier: {album_identifier}")
    
    # Connect to SmugMug to resolve
    client = SmugMugClient(
        api_key=config.get("smugmug.api_key"),
        api_secret=config.get("smugmug.api_secret"),
        access_token=config.get("smugmug.user_token"),
        access_token_secret=config.get("smugmug.user_secret")
    )
    
    album_key = client.resolve_album_key(album_identifier, node_id)
    print(f"✓ Resolved to album key: {album_key}")
    print()
    
    return album_key, album_identifier


def print_banner() -> None:
    """Print smugVision banner."""
    print()
    print("=" * 70)
    print("  smugVision - AI-Powered Photo Metadata Generation")
    print(f"  Version {__version__}")
    print("=" * 70)
    print()


def print_summary(stats, dry_run: bool) -> None:
    """Print processing summary.
    
    Args:
        stats: BatchProcessingStats object
        dry_run: Whether this was a dry run
    """
    print()
    print("=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print()
    
    # Basic stats
    print(f"Total images:     {stats.total_images}")
    print(f"Processed:        {stats.processed} ✓")
    
    if stats.skipped > 0:
        print(f"Skipped:          {stats.skipped} (already tagged)")
    
    if stats.errors > 0:
        print(f"Errors:           {stats.errors} ✗")
    
    print(f"Processing time:  {stats.total_time:.1f}s")
    
    if stats.total_images > 0:
        avg_time = stats.total_time / stats.total_images
        print(f"Avg per image:    {avg_time:.1f}s")
    
    print()
    
    # Success/failure message
    if stats.errors > 0:
        print("⚠️  Some images failed to process. Check logs for details.")
    elif stats.processed == 0 and stats.skipped > 0:
        print("ℹ️  All images already processed. Use --force-reprocess to reprocess.")
    elif dry_run:
        print("✓ Dry run complete! No changes were made to SmugMug.")
        print("  Remove --dry-run to update SmugMug with generated metadata.")
    else:
        print("✓ Processing complete! Metadata has been updated in SmugMug.")
    
    print()


def main() -> int:
    """Main entry point for smugVision CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    args = parse_arguments()
    
    # Setup logging (before banner so initialization logs don't appear)
    if not args.quiet:
        setup_logging(args.verbose)
    else:
        logging.basicConfig(level=logging.ERROR)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Print banner
        if not args.quiet:
            print_banner()
        
        # Load configuration
        if not args.quiet:
            print("Loading configuration...")
        
        if args.config:
            config = ConfigManager.load(config_path=args.config, interactive=False)
        else:
            config = ConfigManager.load(interactive=False)
        
        if not args.quiet:
            print(f"✓ Configuration loaded from: {config.config_path}")
            print()
        
        # Resolve album key
        album_key, album_name = resolve_album_key(args, config)
        
        # Initialize processor
        if not args.quiet:
            print("Initializing processor...")
        
        processor = ImageProcessor(
            config=config,
            dry_run=args.dry_run
        )
        
        if not args.quiet:
            print(f"✓ Vision model: {processor.vision.model_name}")
            if processor.face_recognizer:
                face_count = len(processor.face_recognizer.reference_faces)
                print(f"✓ Face recognition: {face_count} person(s) loaded")
            else:
                print("ℹ️  Face recognition: disabled")
            
            if args.dry_run:
                print("⚠️  DRY RUN MODE - No changes will be made to SmugMug")
            
            print()
            print("-" * 70)
            print(f"Processing album: {album_key}")
            if album_name:
                print(f"Album name: {album_name}")
            print("-" * 70)
            print()
        
        # Process album
        stats = processor.process_album(
            album_key=album_key,
            force_reprocess=args.force_reprocess,
            skip_videos=not args.include_videos
        )
        
        # Print summary
        if not args.quiet:
            print_summary(stats, args.dry_run)
        
        # Return appropriate exit code
        if stats.errors > 0:
            return 1
        
        return 0
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        if not args.quiet:
            print()
            print(f"✗ Configuration Error: {e}")
            print()
            print("Please check your configuration file or run setup:")
            print("  python -m smugvision.config.manager --setup")
        return 2
        
    except SmugMugError as e:
        logger.error(f"SmugMug error: {e}", exc_info=args.verbose)
        if not args.quiet:
            print()
            print(f"✗ SmugMug Error: {e}")
            print()
            print("Troubleshooting:")
            print("  - Verify your SmugMug API credentials in config.yaml")
            print("  - Check that the album key or URL is correct")
            print("  - Ensure you have access to the album")
        return 3
        
    except KeyboardInterrupt:
        if not args.quiet:
            print()
            print()
            print("Processing interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if not args.quiet:
            print()
            print(f"✗ Unexpected Error: {e}")
            print()
            if args.verbose:
                import traceback
                traceback.print_exc()
            else:
                print("Run with --verbose for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())


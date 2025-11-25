#!/usr/bin/env python3
"""Test script for ImageProcessor.

This script tests the image processing pipeline with a SmugMug album.

Usage:
    python test_processor.py <album_key>
    python test_processor.py --url "https://site.smugmug.com/.../album-name"
    python test_processor.py <album_key> --dry-run
    python test_processor.py <album_key> --force-reprocess
"""

import sys
import logging
import argparse
import re
from urllib.parse import urlparse

from smugvision.config import ConfigManager
from smugvision.processing import ImageProcessor
from smugvision.smugmug import SmugMugClient
from smugvision.smugmug.exceptions import SmugMugError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Test image processor."""
    parser = argparse.ArgumentParser(
        description="Test ImageProcessor with SmugMug album",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "album_key",
        nargs="?",
        help="SmugMug album key"
    )
    parser.add_argument(
        "--url",
        help="SmugMug album URL (auto-extracts album key)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview processing without updating SmugMug"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess images even if already tagged"
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        default=True,
        help="Skip video files (default: True)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine album key and node ID
    album_key = None
    node_id = None
    album_identifier = None
    
    if args.url:
        # Extract node ID and album name from URL
        node_match = re.search(r'/n-([a-zA-Z0-9]+)', args.url)
        if node_match:
            node_id = node_match.group(1)
        
        # Get album name from URL
        path = urlparse(args.url).path
        parts = [p for p in path.split('/') if p and not p.startswith('n-')]
        if parts:
            album_identifier = parts[-1]
        
        if not node_id or not album_identifier:
            print("Error: Could not extract node ID and album name from URL")
            print(f"URL: {args.url}")
            print("\nExpected format: ...n-XXXXX/album-name")
            sys.exit(1)
        
        print(f"Extracted from URL:")
        print(f"  Node ID: {node_id}")
        print(f"  Album name: {album_identifier}")
        print()
        
    elif args.album_key:
        album_key = args.album_key
    else:
        parser.print_help()
        sys.exit(1)
    
    print("=" * 70)
    print("smugVision Image Processor Test")
    print("=" * 70)
    print()
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = ConfigManager.load(interactive=False)
        print(f"✓ Configuration loaded")
        print()
        
        # If we have node_id and album_identifier, resolve to album key
        if node_id and album_identifier:
            print("Connecting to SmugMug to resolve album...")
            client = SmugMugClient(
                api_key=config.get("smugmug.api_key"),
                api_secret=config.get("smugmug.api_secret"),
                access_token=config.get("smugmug.user_token"),
                access_token_secret=config.get("smugmug.user_secret")
            )
            
            try:
                album_key = client.resolve_album_key(album_identifier, node_id)
                print(f"✓ Resolved to album key: {album_key}")
                print()
            except SmugMugError as e:
                print(f"\n✗ Error resolving album: {e}")
                print("\nTroubleshooting:")
                print("  - Verify the node ID is correct")
                print("  - Check the exact album name spelling")
                sys.exit(1)
        
        # Initialize processor
        print("Initializing processor...")
        processor = ImageProcessor(
            config=config,
            dry_run=args.dry_run
        )
        print(f"✓ Processor initialized")
        print(f"  Vision model: {processor.vision.model_name}")
        print(f"  Face recognition: {'Enabled' if processor.face_recognizer else 'Disabled'}")
        print(f"  Dry run: {args.dry_run}")
        print()
        
        # Process album
        print(f"Processing album: {album_key}")
        print("-" * 70)
        print()
        
        stats = processor.process_album(
            album_key=album_key,
            force_reprocess=args.force_reprocess,
            skip_videos=args.skip_videos
        )
        
        # Display results
        print()
        print("=" * 70)
        print("Processing Complete")
        print("=" * 70)
        print(f"Total images:    {stats.total_images}")
        print(f"Processed:       {stats.processed}")
        print(f"Skipped:         {stats.skipped}")
        print(f"Errors:          {stats.errors}")
        print(f"Total time:      {stats.total_time:.1f}s")
        
        if stats.total_images > 0:
            avg_time = stats.total_time / stats.total_images
            print(f"Avg time/image:  {avg_time:.1f}s")
        
        print()
        
        # Show detailed results if requested
        if args.verbose and stats.results:
            print("Detailed Results:")
            print("-" * 70)
            for result in stats.results:
                status = "✓" if result.success else "○" if result.skipped else "✗"
                print(f"{status} {result.filename}")
                if result.success:
                    print(f"    Caption: {'Yes' if result.caption_generated else 'No'}")
                    print(f"    Tags: {result.tags_generated}")
                    print(f"    Faces: {result.faces_detected}")
                    print(f"    Time: {result.processing_time:.1f}s")
                elif result.error:
                    print(f"    Error: {result.error}")
                print()
        
        # Exit with appropriate code
        if stats.errors > 0:
            sys.exit(1)
        
    except SmugMugError as e:
        logger.error(f"SmugMug error: {e}", exc_info=True)
        print(f"\n✗ SmugMug Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


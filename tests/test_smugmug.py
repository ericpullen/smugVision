#!/usr/bin/env python3
"""Test script for SmugMug API client.

This script connects to SmugMug and lists images from a specified album,
showing their current captions and tags.

Usage:
    # Using album key (direct)
    python test_smugmug.py <album_key>
    
    # Using album name or URL name (with node ID)
    python test_smugmug.py --node <node_id> --name "Album Name"
    
    # Using SmugMug URL
    python test_smugmug.py --url "https://site.smugmug.com/.../n-ABC123/album-name"
    
Examples:
    python test_smugmug.py abc123
    python test_smugmug.py --node whBRZ3 --name "Grand Finale"
    python test_smugmug.py --url "https://yoursite.smugmug.com/.../n-ABC123/album"
"""

import sys
import logging
import argparse
import re
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from smugvision.config import ConfigManager
from smugvision.smugmug import SmugMugClient
from smugvision.smugmug.exceptions import SmugMugError
from smugvision.cache import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_node_from_url(url: str) -> str:
    """Extract node ID from SmugMug URL.
    
    Args:
        url: SmugMug URL
        
    Returns:
        Node ID or None
    """
    # Look for n-XXXXX pattern
    match = re.search(r'/n-([a-zA-Z0-9]+)', url)
    if match:
        return match.group(1)
    return None


def extract_album_name_from_url(url: str) -> str:
    """Extract album name from SmugMug URL.
    
    Args:
        url: SmugMug URL
        
    Returns:
        Album name or None
    """
    # Get the last part of the path
    path = urlparse(url).path
    parts = [p for p in path.split('/') if p and not p.startswith('n-')]
    if parts:
        return parts[-1]
    return None


def main():
    """Test SmugMug API client."""
    parser = argparse.ArgumentParser(
        description="Test SmugMug API by listing album images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct album key
  python test_smugmug.py abc123
  
  # Search by name under a node
  python test_smugmug.py --node whBRZ3 --name "Grand Finale"
  
  # From URL (auto-extracts node and name)
  python test_smugmug.py --url "https://site.smugmug.com/.../n-ABC123/album-name"
  
  # List all albums under a folder
  python test_smugmug.py --url "https://site.smugmug.com/.../n-ABC123" --list
  python test_smugmug.py --node whBRZ3 --list
"""
    )
    
    parser.add_argument(
        "album_identifier",
        nargs='?',
        help="Album key (e.g., abc123)"
    )
    parser.add_argument(
        "--node",
        help="Node ID to search under (e.g., whBRZ3)"
    )
    parser.add_argument(
        "--name",
        help="Album name to search for"
    )
    parser.add_argument(
        "--url",
        help="SmugMug album URL (auto-extracts node and name)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all albums under a node/folder (use with --node or --url)"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Download images to cache (use with album identifier)"
    )
    parser.add_argument(
        "--size",
        default="Medium",
        choices=["Thumb", "Small", "Medium", "Large", "XLarge", "X2Large", "X3Large", "Original"],
        help="Image size to download (default: Medium)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if images are already cached"
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Include video files in downloads (default: skip videos)"
    )
    
    args = parser.parse_args()
    
    # Determine what we're looking for
    album_identifier = None
    node_id = None
    
    if args.url:
        # Extract from URL
        extracted_node_id = extract_node_from_url(args.url)
        album_name = extract_album_name_from_url(args.url)
        
        # For --list mode, we only need node_id
        if args.list:
            if extracted_node_id:
                # Node ID was in URL
                node_id = extracted_node_id
                print(f"Extracted from URL:")
                print(f"  Node ID: {node_id}")
                print()
            else:
                # No node ID in URL, try to resolve by path
                # Parse path from URL
                parsed_url = urlparse(args.url)
                path = parsed_url.path.strip('/')
                
                if not path:
                    print("Error: Could not extract path from URL")
                    print(f"URL: {args.url}")
                    sys.exit(1)
                
                print(f"No node ID in URL, resolving by path: {path}")
                print()
                
                # Will resolve after authentication
                album_identifier = None
        else:
            # For album viewing, we need both node_id and album_name
            if not extracted_node_id or not album_name:
                print("Error: Could not extract node ID and album name from URL")
                print(f"URL: {args.url}")
                print("\nExpected format: ...n-XXXXX/album-name")
                sys.exit(1)
            
            node_id = extracted_node_id
            album_identifier = album_name
            print(f"Extracted from URL:")
            print(f"  Node ID: {node_id}")
            print(f"  Album name: {album_name}")
            print()
        
    elif args.node and args.name:
        # Node + name search
        node_id = args.node
        album_identifier = args.name
        
    elif args.album_identifier:
        # Direct album key
        album_identifier = args.album_identifier
        
    else:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        print("=" * 70)
        print("smugVision SmugMug API Test")
        print("=" * 70)
        print()
        
        print("Loading configuration...")
        config = ConfigManager.load(interactive=False)
        print(f"✓ Configuration loaded from: {config.config_path}")
        print()
        
        # Initialize SmugMug client
        print("Connecting to SmugMug API...")
        client = SmugMugClient(
            api_key=config.get("smugmug.api_key"),
            api_secret=config.get("smugmug.api_secret"),
            access_token=config.get("smugmug.user_token"),
            access_token_secret=config.get("smugmug.user_secret")
        )
        print("✓ Successfully authenticated with SmugMug")
        print()
        
        # If list mode, show all albums under node
        if args.list:
            if not node_id:
                # Try to resolve by path if URL was provided
                if args.url:
                    parsed_url = urlparse(args.url)
                    path = parsed_url.path.strip('/')
                    
                    if path:
                        print(f"Resolving path to node ID...")
                        try:
                            node_id = client.find_node_by_path(path)
                            if node_id:
                                print(f"✓ Resolved to node ID: {node_id}")
                                print()
                            else:
                                print(f"Error: Could not find node for path: {path}")
                                print("\nTroubleshooting:")
                                print("  - Check that the path exists in your SmugMug")
                                print("  - Try using a URL with n-XXXXX node ID")
                                sys.exit(1)
                        except Exception as e:
                            print(f"Error resolving path: {e}")
                            sys.exit(1)
                
                if not node_id:
                    print("Error: --list requires --node or --url")
                    sys.exit(1)
            
            print(f"Listing all albums under node: {node_id}")
            print("=" * 70)
            print()
            
            try:
                # Get all children recursively
                def get_all_albums(current_node_id: str, prefix: str = "", depth: int = 0, max_depth: int = 5) -> List[tuple]:
                    """Recursively get all albums under a node."""
                    if depth > max_depth:
                        return []
                    
                    albums = []
                    try:
                        children = client.get_node_children(current_node_id)
                    except Exception as e:
                        logger.warning(f"Could not access node {current_node_id}: {e}")
                        return []
                    
                    folders = []
                    for child in children:
                        child_type = child.get("Type")
                        child_name = child.get("Name", "Unnamed")
                        
                        if child_type == "Album":
                            # Extract album key
                            uris = child.get("Uris", {})
                            album_uri = uris.get("Album", {})
                            album_key = None
                            if album_uri:
                                uri = album_uri.get("Uri", "")
                                if "/album/" in uri:
                                    album_key = uri.split("/album/")[-1]
                            
                            if album_key:
                                albums.append((prefix + child_name, album_key, depth))
                        
                        elif child_type == "Folder":
                            folder_node_id = child.get("NodeID")
                            if folder_node_id:
                                folders.append((child_name, folder_node_id))
                    
                    # Process subfolders
                    for folder_name, folder_node_id in folders:
                        folder_prefix = prefix + folder_name + " / "
                        albums.extend(get_all_albums(folder_node_id, folder_prefix, depth + 1, max_depth))
                    
                    return albums
                
                all_albums = get_all_albums(node_id)
                
                if not all_albums:
                    print("No albums found under this node.")
                    return
                
                print(f"Found {len(all_albums)} album(s):")
                print("-" * 70)
                print()
                
                # Group by depth/folder
                current_depth = -1
                for album_name, album_key, depth in all_albums:
                    if depth != current_depth:
                        if current_depth >= 0:
                            print()
                        current_depth = depth
                    
                    # Show album with indentation
                    indent = "  " * depth
                    print(f"{indent}📷 {album_name}")
                    print(f"{indent}   Album Key: {album_key}")
                    print(f"{indent}   Test with: python test_smugmug.py {album_key}")
                    print()
                
                print("=" * 70)
                print(f"Total: {len(all_albums)} album(s)")
                print("=" * 70)
                
            except SmugMugError as e:
                print(f"\nError listing albums: {e}")
                sys.exit(1)
            
            return
        
        # Resolve album identifier to album key
        if node_id:
            print(f"Searching for album '{album_identifier}' under node {node_id}...")
            try:
                album_key = client.resolve_album_key(album_identifier, node_id)
                print(f"✓ Resolved to album key: {album_key}")
            except SmugMugError as e:
                print(f"\nError resolving album: {e}")
                print("\nTroubleshooting:")
                print("  - Verify the node ID is correct")
                print("  - Try using find_album_key.py to list available albums")
                print("  - Check the exact album name spelling")
                sys.exit(1)
        else:
            album_key = album_identifier
        
        print()
        
        # Get album details
        print(f"Fetching album: {album_key}")
        album = client.get_album(album_key)
        
        # If cache mode, download images
        if args.cache:
            print()
            print("=" * 70)
            print("Downloading Images to Cache")
            print("=" * 70)
            print()
            
            # Initialize cache manager
            cache_dir = config.get("cache.directory")
            preserve_structure = config.get("cache.preserve_structure", True)
            
            cache_manager = CacheManager(cache_dir, preserve_structure)
            
            # Determine folder path for cache organization
            folder_path = None
            if preserve_structure and args.url:
                # Extract folder path from URL
                parsed_url = urlparse(args.url)
                url_path = parsed_url.path.strip('/')
                # Remove the album name from the end
                path_parts = [p for p in url_path.split('/') if p and not p.startswith('n-')]
                if len(path_parts) > 1:
                    folder_path = '/'.join(path_parts[:-1])
            
            # Get cache directory for this album
            album_cache_dir = cache_manager.get_album_cache_dir(album.name, folder_path)
            
            print(f"Cache directory: {album_cache_dir}")
            print(f"Image size: {args.size}")
            print(f"Skip existing: {not args.force}")
            print()
            
            # Progress callback
            def progress_callback(current, total, image):
                percent = (current / total) * 100
                print(f"  [{current}/{total}] ({percent:.1f}%) Downloading: {image.file_name}")
            
            try:
                downloaded_paths = client.download_album_images(
                    album_key=album_key,
                    destination=str(album_cache_dir),
                    size=args.size,
                    skip_if_exists=not args.force,
                    skip_videos=not args.include_videos,
                    progress_callback=progress_callback
                )
                
                print()
                print("=" * 70)
                print("Download Summary")
                print("=" * 70)
                print(f"Downloaded: {len(downloaded_paths)} new files")
                
                # Get cache stats
                stats = cache_manager.get_cache_stats()
                print(f"Cache size: {stats['total_size_mb']} MB")
                print(f"Total cached files: {stats['file_count']}")
                print(f"Total cached albums: {stats['album_count']}")
                print()
                
                # Don't show individual images if we downloaded
                print("Use without --cache to view image details")
                return
                
            except SmugMugError as e:
                print(f"\nError downloading images: {e}")
                sys.exit(1)
        print()
        print("=" * 70)
        print(f"Album: {album.name}")
        print("=" * 70)
        if album.description:
            print(f"Description: {album.description}")
        print(f"Images: {album.image_count}")
        print(f"Web URL: {album.web_uri}")
        print()
        
        # Get album images
        print("Fetching images from album...")
        images = client.get_album_images(album_key)
        print(f"✓ Retrieved {len(images)} images")
        print()
        
        if not images:
            print("No images found in this album.")
            return
        
        # Display image details
        print("=" * 70)
        print("Images in Album")
        print("=" * 70)
        print()
        
        for i, image in enumerate(images, 1):
            item_type = "Video" if image.is_video else "Image"
            print(f"{item_type} {i} of {len(images)}")
            print("-" * 70)
            print(f"  Filename:    {image.file_name}")
            print(f"  Image Key:   {image.image_key}")
            print(f"  Format:      {image.format}")
            print(f"  Type:        {'Video' if image.is_video else 'Photo'}")
            
            # Caption
            if image.caption:
                # Wrap long captions
                caption_lines = []
                current_line = ""
                for word in image.caption.split():
                    if len(current_line) + len(word) + 1 <= 60:
                        current_line += (word + " ")
                    else:
                        caption_lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    caption_lines.append(current_line.strip())
                
                print(f"  Caption:     {caption_lines[0]}")
                for line in caption_lines[1:]:
                    print(f"               {line}")
            else:
                print(f"  Caption:     (none)")
            
            # Keywords/Tags
            if image.keywords:
                # Wrap long tag lists
                tags_str = ", ".join(image.keywords)
                if len(tags_str) <= 60:
                    print(f"  Tags:        {tags_str}")
                else:
                    # Wrap tags
                    tag_lines = []
                    current_line = ""
                    for tag in image.keywords:
                        test_line = f"{current_line}, {tag}" if current_line else tag
                        if len(test_line) <= 60:
                            current_line = test_line
                        else:
                            if current_line:
                                tag_lines.append(current_line)
                            current_line = tag
                    if current_line:
                        tag_lines.append(current_line)
                    
                    print(f"  Tags:        {tag_lines[0]}")
                    for line in tag_lines[1:]:
                        print(f"               {line}")
                print(f"  Tag Count:   {len(image.keywords)}")
            else:
                print(f"  Tags:        (none)")
            
            # Check for smugvision marker
            marker_tag = config.get("processing.marker_tag", "smugvision")
            has_marker = image.has_marker_tag(marker_tag)
            if has_marker:
                print(f"  Processed:   ✓ Yes (has '{marker_tag}' tag)")
            else:
                print(f"  Processed:   ✗ No (missing '{marker_tag}' tag)")
            
            # Other metadata
            if image.title:
                print(f"  Title:       {image.title}")
            if image.date:
                print(f"  Date Taken:  {image.date}")
            if image.uploaded:
                print(f"  Uploaded:    {image.uploaded}")
            
            # Web URL
            if image.web_uri:
                print(f"  Web URL:     {image.web_uri}")
            
            print()
        
        # Summary
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Total Images:           {len(images)}")
        
        # Count images with captions/tags
        with_captions = sum(1 for img in images if img.caption)
        with_tags = sum(1 for img in images if img.keywords)
        marker_tag = config.get("processing.marker_tag", "smugvision")
        with_marker = sum(1 for img in images if img.has_marker_tag(marker_tag))
        
        print(f"With Captions:          {with_captions} ({with_captions/len(images)*100:.1f}%)")
        print(f"With Tags:              {with_tags} ({with_tags/len(images)*100:.1f}%)")
        print(f"Already Processed:      {with_marker} ({with_marker/len(images)*100:.1f}%)")
        print()
        
        # Show which images could be processed
        unprocessed = [img for img in images if not img.has_marker_tag(marker_tag)]
        if unprocessed:
            print(f"Ready to Process:       {len(unprocessed)} images")
            print()
            print("Images ready for processing:")
            for img in unprocessed[:5]:  # Show first 5
                print(f"  - {img.file_name}")
            if len(unprocessed) > 5:
                print(f"  ... and {len(unprocessed) - 5} more")
        else:
            print("All images have been processed!")
        
        print()
        print("=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        
    except SmugMugError as e:
        print(f"\nSmugMug Error: {e}", file=sys.stderr)
        logger.exception("SmugMug API error")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        logger.exception("Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


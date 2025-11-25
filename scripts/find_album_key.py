#!/usr/bin/env python3
"""Helper script to find album keys from node IDs or URLs.

This script helps you discover album keys by:
1. Listing children of a node (folders and albums)
2. Finding albums by name

Usage:
    # List all albums under a node
    python find_album_key.py --node whBRZ3
    
    # Search for album by name
    python find_album_key.py --node whBRZ3 --search "Grand Finale"
    
    # From a URL
    python find_album_key.py --url "https://yoursite.smugmug.com/Gallery/Year/n-ABC123/..."
"""

import sys
import argparse
import logging
import re
from urllib.parse import urlparse

from smugvision.config import ConfigManager
from smugvision.smugmug import SmugMugClient
from smugvision.smugmug.exceptions import SmugMugError

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def extract_node_from_url(url: str) -> str:
    """Extract node ID from SmugMug URL."""
    # Look for n-XXXXX pattern
    match = re.search(r'/n-([a-zA-Z0-9]+)', url)
    if match:
        return match.group(1)
    return None


def list_node_children(client: SmugMugClient, node_id: str, search_term: str = None):
    """List all children (albums and folders) under a node."""
    print(f"Fetching contents of node: {node_id}")
    print("=" * 70)
    
    try:
        # Get node children
        endpoint = f"/node/{node_id}!children"
        response = client._request("GET", endpoint)
        
        children = response.get("Response", {}).get("Node", [])
        
        if not children:
            print("No albums or folders found under this node.")
            return
        
        albums = []
        folders = []
        
        for child in children:
            node_type = child.get("Type")
            name = child.get("Name", "Unnamed")
            url_name = child.get("UrlName", "")
            node_id = child.get("NodeID", "")
            
            if node_type == "Album":
                # Get album key
                uris = child.get("Uris", {})
                album_uri = uris.get("Album", {})
                album_key = None
                if album_uri:
                    # Extract album key from URI
                    uri = album_uri.get("Uri", "")
                    if "/album/" in uri:
                        album_key = uri.split("/album/")[-1]
                
                albums.append({
                    "name": name,
                    "url_name": url_name,
                    "album_key": album_key,
                    "node_id": node_id
                })
            elif node_type == "Folder":
                folders.append({
                    "name": name,
                    "node_id": node_id
                })
        
        # Display folders
        if folders:
            print(f"\nFolders ({len(folders)}):")
            print("-" * 70)
            for folder in folders:
                print(f"  📁 {folder['name']}")
                print(f"     Node ID: {folder['node_id']}")
                print()
        
        # Display albums
        if albums:
            # Filter by search term if provided
            if search_term:
                search_lower = search_term.lower()
                albums = [a for a in albums if search_lower in a['name'].lower()]
                
                if not albums:
                    print(f"\nNo albums found matching: {search_term}")
                    return
                
                print(f"\nAlbums matching '{search_term}' ({len(albums)}):")
            else:
                print(f"\nAlbums ({len(albums)}):")
            
            print("-" * 70)
            
            for album in albums:
                print(f"  📷 {album['name']}")
                if album['album_key']:
                    print(f"     Album Key: {album['album_key']}")
                    print(f"     Test with: python test_smugmug.py {album['album_key']}")
                else:
                    print(f"     Album Key: (not found - try node {album['node_id']})")
                print()
        
        if not albums and not folders:
            print("No content found.")
        
    except SmugMugError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure the node ID is correct")
        print("  - Try using the parent folder's node ID")
        print("  - Check that you have access to this content")
        sys.exit(1)


def main():
    """Find album keys from nodes or URLs."""
    parser = argparse.ArgumentParser(
        description="Find SmugMug album keys from node IDs or URLs"
    )
    parser.add_argument(
        "--node",
        help="Node ID (e.g., whBRZ3)"
    )
    parser.add_argument(
        "--url",
        help="SmugMug URL to extract node from"
    )
    parser.add_argument(
        "--search",
        help="Search for albums by name"
    )
    
    args = parser.parse_args()
    
    # Get node ID
    node_id = None
    if args.url:
        node_id = extract_node_from_url(args.url)
        if not node_id:
            print("Error: Could not extract node ID from URL")
            print(f"URL: {args.url}")
            print("\nExpected format: ...n-XXXXX...")
            sys.exit(1)
        print(f"Extracted node ID from URL: {node_id}\n")
    elif args.node:
        node_id = args.node
    else:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load config and authenticate
        config = ConfigManager.load(interactive=False)
        
        client = SmugMugClient(
            api_key=config.get("smugmug.api_key"),
            api_secret=config.get("smugmug.api_secret"),
            access_token=config.get("smugmug.user_token"),
            access_token_secret=config.get("smugmug.user_secret")
        )
        
        # List children
        list_node_children(client, node_id, args.search)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


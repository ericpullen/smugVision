#!/bin/bash
# Example workflow for smugVision - Demonstrates typical usage

set -e  # Exit on error

echo "========================================================================"
echo "smugVision Example Workflow"
echo "========================================================================"
echo ""

# Check if URL was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <smugmug_album_url>"
    echo ""
    echo "Example:"
    echo "  $0 'https://yoursite.smugmug.com/path/to/n-XXXXX/album-name'"
    echo ""
    exit 1
fi

ALBUM_URL="$1"

echo "Processing album: $ALBUM_URL"
echo ""

# Step 1: Preview what would be generated (dry run)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Preview Processing (Dry Run)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will show what captions and tags would be generated without"
echo "actually updating SmugMug."
echo ""
read -p "Press Enter to continue..."
echo ""

python -m smugvision --url "$ALBUM_URL" --dry-run

echo ""
echo ""

# Step 2: Ask user if they want to proceed
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Process Album (Update SmugMug)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Do you want to proceed with updating SmugMug?"
read -p "Type 'yes' to continue, anything else to cancel: " CONFIRM
echo ""

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled. No changes were made to SmugMug."
    exit 0
fi

# Step 3: Process for real
echo "Processing album and updating SmugMug..."
echo ""

python -m smugvision --url "$ALBUM_URL"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Processing Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Your photos on SmugMug have been updated with:"
echo "  - AI-generated captions"
echo "  - Relevant keyword tags"
echo "  - Location information (if available)"
echo "  - Identified people (if face recognition is enabled)"
echo ""
echo "Visit SmugMug to see the results!"
echo ""


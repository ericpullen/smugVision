#!/usr/bin/env python3
"""OAuth 1.0a authorization script for SmugMug.

This script helps you obtain user access tokens (user_token and user_secret)
for your SmugMug account. You only need to run this once.

Prerequisites:
1. SmugMug API key and secret from https://api.smugmug.com/api/developer/apply
2. requests-oauthlib library: pip install requests-oauthlib

Usage:
    python get_smugmug_tokens.py

This will:
1. Prompt you for your API key and secret
2. Get a request token from SmugMug
3. Open your browser for authorization
4. Exchange the authorized request token for access tokens
5. Display your user_token and user_secret to add to config.yaml
"""

import sys
import webbrowser
from urllib.parse import parse_qs, urlparse

try:
    from requests_oauthlib import OAuth1Session
except ImportError:
    print("Error: requests-oauthlib is required")
    print("Install with: pip install requests-oauthlib")
    sys.exit(1)


# SmugMug OAuth endpoints
REQUEST_TOKEN_URL = "https://api.smugmug.com/services/oauth/1.0a/getRequestToken"
AUTHORIZE_URL = "https://api.smugmug.com/services/oauth/1.0a/authorize"
ACCESS_TOKEN_URL = "https://api.smugmug.com/services/oauth/1.0a/getAccessToken"


def get_access_tokens():
    """Walk through OAuth 1.0a flow to get access tokens."""
    
    print("=" * 70)
    print("SmugMug OAuth 1.0a Authorization")
    print("=" * 70)
    print()
    print("This script will help you get OAuth access tokens for your SmugMug account.")
    print()
    
    # Step 1: Get API credentials
    print("Step 1: Enter your SmugMug API credentials")
    print("(Get these from https://api.smugmug.com/api/developer/apply)")
    print()
    
    api_key = input("API Key (Consumer Key): ").strip()
    if not api_key:
        print("Error: API Key is required")
        sys.exit(1)
    
    api_secret = input("API Secret (Consumer Secret): ").strip()
    if not api_secret:
        print("Error: API Secret is required")
        sys.exit(1)
    
    print()
    print("=" * 70)
    
    try:
        # Step 2: Get request token
        print("\nStep 2: Getting request token from SmugMug...")
        
        oauth = OAuth1Session(
            api_key,
            client_secret=api_secret,
            callback_uri='oob'  # Out of band - manual code entry
        )
        
        response = oauth.fetch_request_token(REQUEST_TOKEN_URL)
        request_token = response.get('oauth_token')
        request_token_secret = response.get('oauth_token_secret')
        
        if not request_token:
            print("Error: Failed to get request token")
            sys.exit(1)
        
        print("✓ Got request token")
        
        # Step 3: User authorization
        print()
        print("=" * 70)
        print("\nStep 3: Authorize access to your SmugMug account")
        print()
        print("Your browser will open to authorize smugVision.")
        print("After authorizing, you'll receive a 6-digit verification code.")
        print()
        input("Press Enter to open browser...")
        
        # Build authorization URL with access and permissions
        auth_url = f"{AUTHORIZE_URL}?oauth_token={request_token}&Access=Full&Permissions=Modify"
        
        # Open browser
        webbrowser.open(auth_url)
        
        print()
        print("Browser opened. Please:")
        print("1. Log in to SmugMug if prompted")
        print("2. Click 'Authorize' to grant access")
        print("3. Copy the 6-digit verification code shown")
        print()
        
        verifier = input("Enter the 6-digit verification code: ").strip()
        if not verifier:
            print("Error: Verification code is required")
            sys.exit(1)
        
        # Step 4: Exchange for access token
        print()
        print("=" * 70)
        print("\nStep 4: Exchanging for access tokens...")
        
        oauth = OAuth1Session(
            api_key,
            client_secret=api_secret,
            resource_owner_key=request_token,
            resource_owner_secret=request_token_secret,
            verifier=verifier
        )
        
        access_response = oauth.fetch_access_token(ACCESS_TOKEN_URL)
        
        user_token = access_response.get('oauth_token')
        user_secret = access_response.get('oauth_token_secret')
        
        if not user_token or not user_secret:
            print("Error: Failed to get access tokens")
            print("Response:", access_response)
            sys.exit(1)
        
        print("✓ Successfully obtained access tokens!")
        
        # Display results
        print()
        print("=" * 70)
        print("SUCCESS! Your OAuth Tokens")
        print("=" * 70)
        print()
        print("Add these to your ~/.smugvision/config.yaml file:")
        print()
        print("smugmug:")
        print(f"  api_key: \"{api_key}\"")
        print(f"  api_secret: \"{api_secret}\"")
        print(f"  user_token: \"{user_token}\"")
        print(f"  user_secret: \"{user_secret}\"")
        print()
        print("=" * 70)
        print()
        print("Save these tokens! You only need to do this once.")
        print("These tokens allow smugVision to access your SmugMug account.")
        print()
        
    except Exception as e:
        print(f"\nError during OAuth flow: {e}")
        print()
        print("Common issues:")
        print("- Invalid API key or secret")
        print("- Network connection problems")
        print("- Incorrect verification code")
        print("- API key not yet approved by SmugMug")
        sys.exit(1)


if __name__ == "__main__":
    try:
        get_access_tokens()
    except KeyboardInterrupt:
        print("\n\nAuthorization cancelled.")
        sys.exit(1)


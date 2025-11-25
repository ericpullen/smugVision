# SmugMug API Testing Guide

This guide explains how to test the SmugMug API integration with your account.

## Prerequisites

1. **Install required dependencies:**
   ```bash
   pip install requests-oauthlib
   ```

2. **SmugMug API credentials** (see setup below if you don't have them)

3. **Configuration file** with SmugMug credentials at `~/.smugvision/config.yaml`

## Getting SmugMug API Credentials

SmugMug uses **OAuth 1.0a** for authentication, which requires **two sets of credentials**:

1. **Application Credentials** - Identify your app (smugVision)
2. **User Access Tokens** - Identify which SmugMug user and what permissions

### Step 1: Apply for API Key (Application Credentials)

1. Visit the [SmugMug API Developer Page](https://api.smugmug.com/api/developer/apply)
2. Log in with your SmugMug account
3. Fill out the application form:
   - **Application Name**: smugVision (or your preferred name)
   - **Description**: AI-powered photo metadata generation
   - **Platform**: Desktop/Web Application
4. Submit and wait for approval (usually quick)

After approval, you'll receive:
- ✅ **API Key** (also called Consumer Key)
- ✅ **API Secret** (also called Consumer Secret)

### Step 2: Get OAuth Access Tokens (User Tokens)

The API key/secret identify your **application**, but you also need tokens that identify which **SmugMug user** is using it. This is a separate OAuth authorization step.

#### Option A: Use Our Authorization Script (Recommended)

We've created a script to walk you through the OAuth flow:

```bash
python get_smugmug_tokens.py
```

This will:
1. Ask for your API key and secret (from Step 1)
2. Open your browser to SmugMug's authorization page
3. You authorize smugVision to access your account
4. Get a 6-digit verification code
5. Exchange it for access tokens
6. Display your **user_token** and **user_secret**

#### Option B: Manual OAuth Flow

If you prefer to do it manually:

1. Create an OAuth request token using your API credentials
2. Navigate to: `https://api.smugmug.com/services/oauth/1.0a/authorize?oauth_token={request_token}&Access=Full&Permissions=Modify`
3. Authorize access and get verification code
4. Exchange the verification code for access tokens

(The script in Option A does all this automatically)

#### What You'll Get

After authorization, you'll have:
- ✅ **user_token** (OAuth Access Token)
- ✅ **user_secret** (OAuth Access Token Secret)

**Important:** You only need to do this **once**. Save these tokens - they don't expire unless you revoke them.

### Step 3: Add to Configuration

Edit your `~/.smugvision/config.yaml`:

```yaml
smugmug:
  api_key: "YOUR_API_KEY_HERE"
  api_secret: "YOUR_API_SECRET_HERE"
  user_token: "YOUR_ACCESS_TOKEN_HERE"
  user_secret: "YOUR_ACCESS_TOKEN_SECRET_HERE"
```

## Finding Your Album Key

To test with an album, you need its **Album Key**:

1. Go to any album on your SmugMug site
2. Look at the URL, it will be something like:
   ```
   https://yoursite.smugmug.com/Album-Name/n-AbCdEf
   ```
3. The album key is the part after `n-`: **AbCdEf**

Alternatively, you can find it in the album settings or by using the SmugMug API browser.

## Running the Test

Once you have your credentials configured and an album key:

```bash
python test_smugmug.py <ALBUM_KEY>
```

**Example:**
```bash
python test_smugmug.py AbCdEf
```

## What the Test Does

The test script will:

1. ✅ Load configuration from `~/.smugvision/config.yaml`
2. ✅ Authenticate with SmugMug using OAuth 1.0a
3. ✅ Fetch album details (name, description, image count)
4. ✅ Retrieve all images from the album (handles pagination automatically)
5. ✅ Display each image's:
   - Filename
   - Current caption (if any)
   - Current tags/keywords (if any)
   - Whether it has the "smugvision" marker tag
   - Other metadata (date taken, uploaded, etc.)
6. ✅ Show summary statistics:
   - Total images
   - How many have captions
   - How many have tags
   - How many have been processed already
   - Which images are ready for processing

## Expected Output

```
======================================================================
smugVision SmugMug API Test
======================================================================

Loading configuration...
✓ Configuration loaded from: /Users/you/.smugvision/config.yaml

Connecting to SmugMug API...
✓ Successfully authenticated with SmugMug

Fetching album: AbCdEf

======================================================================
Album: My Test Album
======================================================================
Description: Test album for smugVision
Images: 25
Web URL: https://yoursite.smugmug.com/...

Fetching images from album...
✓ Retrieved 25 images

======================================================================
Images in Album
======================================================================

Image 1 of 25
----------------------------------------------------------------------
  Filename:    IMG_1234.jpg
  Image Key:   xyz123
  Format:      JPG
  Caption:     A beautiful sunset over the Golden Gate Bridge in San
               Francisco, California
  Tags:        sunset, bridge, san francisco, california, landscape
  Tag Count:   5
  Processed:   ✓ Yes (has 'smugvision' tag)
  Date Taken:  2024-11-15T18:30:00Z
  Uploaded:    2024-11-16T10:15:00Z
  Web URL:     https://yoursite.smugmug.com/...

Image 2 of 25
----------------------------------------------------------------------
  Filename:    IMG_1235.jpg
  Image Key:   xyz124
  Format:      JPG
  Caption:     (none)
  Tags:        (none)
  Processed:   ✗ No (missing 'smugvision' tag)
  Date Taken:  2024-11-15T19:00:00Z

...

======================================================================
Summary
======================================================================
Total Images:           25
With Captions:          10 (40.0%)
With Tags:              12 (48.0%)
Already Processed:      8 (32.0%)

Ready to Process:       17 images

Images ready for processing:
  - IMG_1235.jpg
  - IMG_1236.jpg
  - IMG_1237.jpg
  - IMG_1238.jpg
  - IMG_1239.jpg
  ... and 12 more

======================================================================
Test completed successfully!
======================================================================
```

## Troubleshooting

### "Authentication failed" Error

**Problem:** Invalid credentials or OAuth tokens expired.

**Solutions:**
- Double-check all four credentials in config.yaml
- Make sure you're using OAuth 1.0a tokens (not OAuth 2.0)
- Verify your API key is approved and active
- Try re-generating your access tokens

### "Album not found" Error

**Problem:** Invalid album key or album not accessible.

**Solutions:**
- Verify the album key is correct (check the URL)
- Make sure the album belongs to your account
- Check that the album is not private/hidden from API access
- Try a different album to rule out album-specific issues

### "Rate limit exceeded" Error

**Problem:** Too many API requests in a short time.

**Solution:**
- Wait a few minutes before trying again
- SmugMug has rate limits documented in their API docs
- The client will tell you how long to wait

### "Configuration file not found" Error

**Problem:** No config.yaml found.

**Solution:**
```bash
mkdir -p ~/.smugvision
cp config.yaml.example ~/.smugvision/config.yaml
# Edit ~/.smugvision/config.yaml with your credentials
```

### Network/Connection Errors

**Problem:** Cannot reach SmugMug API.

**Solutions:**
- Check your internet connection
- Verify you can access https://api.smugmug.com in a browser
- Check if a firewall is blocking the connection
- Try increasing the timeout in the config

## Next Steps

Once the test is successful, you're ready to:

1. **Process images with AI:**
   - Use the vision model to generate captions and tags
   - Update SmugMug images with new metadata

2. **Batch process entire albums:**
   - The client handles pagination automatically
   - Can process hundreds of images

3. **Integrate with face recognition:**
   - Identify people in photos
   - Include names in captions and tags

## API Documentation

For more details on the SmugMug API v2:
- [SmugMug API Documentation](https://api.smugmug.com/api/v2/doc)
- [API Tutorial](https://api.smugmug.com/api/v2/doc/tutorial)
- [Live API Browser](https://api.smugmug.com/api/v2) (explore while logged in)

## Support

If you encounter issues:

1. Check the log file: `~/.smugvision/smugvision.log`
2. Run with DEBUG logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. Review SmugMug API status page
4. Check GitHub issues for similar problems


# SmugMug OAuth Setup - Visual Guide

## Understanding SmugMug Authentication

SmugMug uses **OAuth 1.0a**, which requires **4 pieces of information** total:

```
┌─────────────────────────────────────────────────────────────┐
│                  SmugMug Authentication                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Application Credentials (from SmugMug Developer Portal) │
│     ├── api_key        (identifies YOUR app)                │
│     └── api_secret     (proves it's YOUR app)               │
│                                                              │
│  2. User Access Tokens (from OAuth authorization flow)      │
│     ├── user_token     (which SmugMug user)                 │
│     └── user_secret    (proves user authorized you)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Why Both Are Needed

Think of it like a building with security:

- **API Key/Secret**: Your company ID badge (identifies smugVision app)
- **User Token/Secret**: Visitor pass for a specific person (your SmugMug account)

Both are required to access the building (SmugMug API).

## Step-by-Step Setup

### Part 1: Get Application Credentials (5 minutes)

```
1. Go to: https://api.smugmug.com/api/developer/apply
   
2. Fill out form:
   ┌──────────────────────────────────────┐
   │ Application Name: smugVision         │
   │ Description: AI photo metadata       │
   │ Platform: Desktop Application        │
   └──────────────────────────────────────┘

3. Submit and wait for approval email

4. You receive:
   ✅ API Key (e.g., "abc123...")
   ✅ API Secret (e.g., "xyz789...")
```

**⚠️ Stop here!** You don't have user tokens yet. You need Part 2.

### Part 2: Get User Access Tokens (2 minutes)

```
Option A: Use our script (easiest)
─────────────────────────────────────

1. Run: python get_smugmug_tokens.py

2. Enter your API Key and Secret from Part 1

3. Browser opens to SmugMug authorization page:
   
   ┌─────────────────────────────────────────┐
   │  smugVision wants to access your        │
   │  SmugMug account                        │
   │                                         │
   │  This will allow smugVision to:        │
   │  • View your photos                    │
   │  • Modify photo metadata               │
   │                                         │
   │  [ Cancel ]  [ Authorize ]             │
   └─────────────────────────────────────────┘

4. Click "Authorize"

5. You get a 6-digit code: 123456

6. Enter code in terminal

7. Script displays:
   ✅ user_token (e.g., "def456...")
   ✅ user_secret (e.g., "ghi789...")


Option B: Manual OAuth (advanced)
──────────────────────────────────

If you're comfortable with OAuth flows, you can:
1. Request a request token
2. Authorize at: https://api.smugmug.com/services/oauth/1.0a/authorize
3. Exchange verification code for access tokens

The script does this automatically for you.
```

### Part 3: Add to Configuration

```
1. Edit: ~/.smugvision/config.yaml

2. Add all 4 credentials:

   smugmug:
     api_key: "abc123..."        # From Part 1
     api_secret: "xyz789..."     # From Part 1
     user_token: "def456..."     # From Part 2
     user_secret: "ghi789..."    # From Part 2

3. Save file

4. Test: python test_smugmug.py <album_key>
```

## Common Questions

### Q: Why do I need user tokens if I have an API key?

**A:** The API key says "this is smugVision app" but doesn't say **which SmugMug user**. The user tokens say "John Doe has authorized smugVision to access his photos."

### Q: Do these tokens expire?

**A:** 
- **API Key/Secret**: No expiration (unless you revoke them)
- **User Token/Secret**: No expiration (unless you revoke or change password)

You only need to get them once!

### Q: Are these tokens secure?

**A:** Yes, but:
- ✅ Store in `~/.smugvision/config.yaml` (not in code)
- ✅ Add `config.yaml` to `.gitignore`
- ✅ Don't share them publicly
- ✅ Use file permissions: `chmod 600 ~/.smugvision/config.yaml`

### Q: Can I use the same tokens on multiple computers?

**A:** Yes! Copy your `config.yaml` to each computer. The tokens work anywhere.

### Q: What if I see "Authentication failed"?

**A:** Check:
1. All 4 credentials are in config.yaml
2. No extra spaces or quotes
3. API key is approved (check email)
4. User tokens were obtained for the correct API key
5. Run `get_smugmug_tokens.py` again if needed

### Q: What permissions do these tokens have?

**A:** The tokens have:
- ✅ Read access (view albums and images)
- ✅ Write access (update captions and keywords)
- ✅ Full access to your account

This is required for smugVision to work. You can revoke access anytime from SmugMug settings.

## Security Best Practices

```
✅ DO:
  - Store credentials in ~/.smugvision/config.yaml
  - Set file permissions: chmod 600 ~/.smugvision/config.yaml
  - Add config.yaml to .gitignore
  - Revoke tokens if compromised

❌ DON'T:
  - Commit credentials to git
  - Share tokens publicly
  - Use tokens from untrusted sources
  - Store in public repositories
```

## Troubleshooting

### "API key not found"
→ Wait for approval email from SmugMug
→ Check spam folder

### "Invalid OAuth signature"
→ Check all 4 credentials are correct
→ No extra spaces in config.yaml
→ User tokens match the API key used

### "Browser doesn't open"
→ Script shows URL, copy to browser manually
→ Or use manual OAuth flow

### "Verification code doesn't work"
→ Make sure it's the 6-digit code from SmugMug
→ Try getting a new code (run script again)
→ Check you authorized the correct app

## Quick Reference

```bash
# 1. Get API key
Visit: https://api.smugmug.com/api/developer/apply

# 2. Get user tokens
python get_smugmug_tokens.py

# 3. Test authentication
python test_smugmug.py <album_key>
```

## Need Help?

1. Check SmugMug API docs: https://api.smugmug.com/api/v2/doc
2. Review OAuth 1.0a spec: https://oauth.net/core/1.0a/
3. Check smugVision logs: ~/.smugvision/smugvision.log


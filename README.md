# YouTube Creator Studio Pro (Desktop-Web Studio) + Universal MCP SSE Server

A YouTube Studio desktop-mode web application and universal **Model Context Protocol (MCP)** server built with Python, FastMCP, and YouTube Data API v3 for managing, optimizing, and uploading videos to YouTube (`shoaibgh473@gmail.com`).

---

## 🤖 Universal MCP SSE Endpoints (Gemini / ChatGPT / Grok)

This service exposes a fully compliant Model Context Protocol (MCP) Server-Sent Events (SSE) endpoint:

- **SSE Transport Route**: `/sse`
- **MCP Alias Route**: `/mcp`
- **Health Check Route**: `GET /health` (returns `200 OK`)
- **CORS Allowed Origin**: `*` (Universal cross-origin access enabled)
- **Live Production URL**: `https://youtube-studio-pro.onrender.com/sse`
- **Live Production Alias**: `https://youtube-studio-pro.onrender.com/mcp`
- **Live Health Endpoint**: `https://youtube-studio-pro.onrender.com/health`

### Connecting to Gemini / ChatGPT / Grok:
In Gemini's Custom MCP Connector, ChatGPT Developer Mode, or Grok, enter:
```
https://youtube-studio-pro.onrender.com/sse
```
(or `https://youtube-studio-pro.onrender.com/mcp`)

### Available MCP Tools:

#### 1. `optimize_video_metadata`
Optimizes YouTube video title, description, and tags for maximum Click-Through Rate (CTR) and search discoverability using Gemini AI (with intelligent SEO algorithmic fallback).
- `title` (string, required): Working title or video topic.
- `description` (string, required): Draft description or topic overview.
- `tags` (array of strings, required): Seed tags or keyword themes.
- `video_url` (string, optional): Video file URL or YouTube link.
- `format_type` (string, optional): `'Short'` (9:16) or `'Long'` (16:9).
- `custom_instructions` (string, optional): Specific tone, niche, or guidance.

#### 2. `upload_to_youtube`
Uploads or schedules a video on YouTube with standard metadata and privacy settings.
- `title` (string, required): Video title (max 100 chars).
- `description` (string, required): Video description (max 5,000 chars).
- `tags` (array of strings, required): List of tags/keywords.
- `video_url` (string, optional): Direct URL or path to video file.
- `privacy_status` (string, optional): `'public'`, `'private'`, or `'unlisted'`. Default is `'private'`.
- `category_id` (string, optional): YouTube category ID (default: `'22'`).
- `made_for_kids` (boolean, optional): COPPA declaration (default: `false`).

---

## 🚀 Status & Local Development

- **Local Runner:** `python app.py` (or `uvicorn app:asgi_app --host 0.0.0.0 --port 8000`)
- **Environment:** Python 3.12 with `mcp>=1.3.0,<2.0.0`, `uvicorn`, `starlette`, `a2wsgi`, `google-api-python-client`, `google-auth-oauthlib`, `flask`
- **Default Port:** `8000` (configurable via `PORT` environment variable)

---

## 📋 Features Included

1. **One-Click Google Login / OAuth 2.0**:
   - Targets `shoaibgh473@gmail.com` with `login_hint`.
   - Scopes: `youtube.upload`, `youtube`, `youtube.force-ssl`, `youtube.readonly`.
   - Auto-refreshes credentials and caches them in `token.json` or `YOUTUBE_TOKEN_JSON`.

2. **Channel Overview Sidebar**:
   - Channel branding (Avatar, Title, Handle).
   - Live Statistics (Subscriber count, Total views, Total videos).
   - Quota tracking meter (~1,600 / 10,000 daily points).

3. **Complete Upload Studio**:
   - Multi-format file selector (MP4, MKV, MOV, WebM).
   - Real-time custom thumbnail selector with instant image preview.
   - Title input with live character counter (100 char limit).
   - Description box with live character counter (5,000 char limit).
   - Interactive Tag Chips (type and press Enter/comma).
   - Privacy status dropdown (`Public`, `Unlisted`, `Private`).
   - Category selector (Gaming, Tech, Education, Entertainment, etc.).
   - Audience toggle (COPPA "Made for Kids" declaration).
   - Two-phase real-time resumable upload progress bar.

4. **MCP Server Integration**:
   - Fully unified ASGI server serving both the web UI and FastMCP SSE endpoint on a single port.
   - CORS enabled for browser and cloud LLM execution.
   - Instant health check for Render container monitoring.

---

## 🛠️ Start Commands

### Local:
```powershell
python app.py
```
Or with uvicorn:
```powershell
uvicorn app:asgi_app --host 0.0.0.0 --port 8000
```

### Render Deployment:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:asgi_app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120`

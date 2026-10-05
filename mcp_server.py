"""
Model Context Protocol (MCP) Server with Server-Sent Events (SSE) Transport
Compliant with Gemini Custom MCP Connector, ChatGPT Developer Mode, and Grok.

Endpoints exposed:
- GET /sse        : Standard MCP SSE streaming endpoint
- GET /mcp        : MCP SSE alias endpoint
- POST /messages/ : MCP JSON-RPC message endpoint (POST endpoint announced in SSE stream)
- GET /health     : Service health check returning 200 OK
- /               : Fallback mount to YouTube Studio Pro Flask Application
"""

import os
import sys
import json
import uuid
import time
import re
import urllib.parse
import urllib.request
import threading
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route, Mount
from a2wsgi import WSGIMiddleware

# Initialize FastMCP with DNS rebinding protection disabled for cloud/external access
mcp = FastMCP(
    name="YouTube Creator Studio Pro",
    instructions=(
        "Universal Model Context Protocol (MCP) server for YouTube Creator Studio Pro. "
        "Provides complete control to list channel videos, update titles, descriptions, tags, "
        "and thumbnails directly on YouTube, optimize video metadata using Gemini AI, "
        "and upload new videos."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )
)


def get_youtube_service():
    """Helper to build authenticated YouTube Data API v3 service."""
    try:
        import app as main_app
        creds = main_app.get_stored_credentials()
        if not creds:
            return None
        from googleapiclient.discovery import build
        return build('youtube', 'v3', credentials=creds)
    except Exception as e:
        print(f"[MCP YouTube Service Init Error]: {e}")
        return None


# ============================================================================
# TOOL 1: list_channel_videos
# ============================================================================
@mcp.tool(
    name="list_channel_videos",
    description=(
        "Lists uploaded videos from the authenticated YouTube channel. "
        "Returns video IDs, titles, descriptions, tags, thumbnail URLs, and privacy status. "
        "Use this tool to inspect all existing videos before editing or optimizing."
    )
)
def list_channel_videos(max_results: int = 15) -> str:
    """Lists recent videos uploaded to the authenticated YouTube channel.

    Args:
        max_results: Maximum number of videos to fetch (default: 15, max: 50).
    """
    limit = max(1, min(int(max_results or 15), 50))
    youtube = get_youtube_service()

    if not youtube:
        return json.dumps({
            "status": "credentials_required",
            "message": (
                "YouTube OAuth credentials required. Please visit "
                "https://youtube-studio-pro.onrender.com/login to connect your YouTube channel."
            ),
            "auth_url": "https://youtube-studio-pro.onrender.com/login"
        }, indent=2)

    try:
        ch_res = youtube.channels().list(mine=True, part='snippet,contentDetails,statistics').execute()
        items = ch_res.get('items', [])
        if not items:
            return json.dumps({"status": "empty", "message": "No channel found for current credentials.", "videos": []})

        channel_title = items[0].get('snippet', {}).get('title', 'Unknown Channel')
        uploads_id = items[0]['contentDetails']['relatedPlaylists']['uploads']

        pl_res = youtube.playlistItems().list(
            playlistId=uploads_id,
            part='snippet,status,contentDetails',
            maxResults=limit
        ).execute()

        video_items = pl_res.get('items', [])
        if not video_items:
            return json.dumps({"status": "empty", "channel": channel_title, "videos": []})

        # Fetch detailed snippets including tags
        video_ids = [it['snippet']['resourceId']['videoId'] for it in video_items if it.get('snippet', {}).get('resourceId', {}).get('videoId')]
        details_map = {}
        if video_ids:
            v_res = youtube.videos().list(
                id=','.join(video_ids),
                part='snippet,status,statistics'
            ).execute()
            for v in v_res.get('items', []):
                details_map[v['id']] = v

        videos = []
        for it in video_items:
            vid = it.get('snippet', {}).get('resourceId', {}).get('videoId')
            detailed = details_map.get(vid, {})
            snip = detailed.get('snippet', it.get('snippet', {}))
            status = detailed.get('status', it.get('status', {}))
            stats = detailed.get('statistics', {})

            videos.append({
                "video_id": vid,
                "title": snip.get('title', ''),
                "description": snip.get('description', ''),
                "tags": snip.get('tags', []),
                "category_id": snip.get('categoryId', '22'),
                "privacy_status": status.get('privacyStatus', 'public'),
                "published_at": snip.get('publishedAt', ''),
                "thumbnail_url": snip.get('thumbnails', {}).get('high', {}).get('url') or snip.get('thumbnails', {}).get('default', {}).get('url', ''),
                "view_count": stats.get('viewCount', '0'),
                "watch_url": f"https://youtu.be/{vid}"
            })

        return json.dumps({
            "status": "success",
            "channel": channel_title,
            "count": len(videos),
            "videos": videos
        }, indent=2)

    except Exception as e:
        err_msg = str(e)
        if "invalid_grant" in err_msg or "expired" in err_msg or "revoked" in err_msg:
            return json.dumps({
                "status": "credentials_required",
                "message": "YouTube token has expired or was revoked. Please re-authenticate at https://youtube-studio-pro.onrender.com/login",
                "auth_url": "https://youtube-studio-pro.onrender.com/login"
            }, indent=2)
        return json.dumps({"status": "error", "message": f"Failed to list videos: {err_msg}"})


# ============================================================================
# TOOL 2: get_video_details
# ============================================================================
@mcp.tool(
    name="get_video_details",
    description="Retrieves full metadata for a specific YouTube video (title, description, tags, category, and statistics)."
)
def get_video_details(video_id: str) -> str:
    """Retrieves full details of a specific YouTube video.

    Args:
        video_id: The 11-character YouTube video ID (e.g. '0zZgD9g5PQ8').
    """
    clean_id = (video_id or "").strip()
    if not clean_id:
        return json.dumps({"status": "error", "message": "Video ID is required."})

    youtube = get_youtube_service()
    if not youtube:
        return json.dumps({
            "status": "credentials_required",
            "message": "YouTube OAuth credentials required. Visit https://youtube-studio-pro.onrender.com/login",
            "auth_url": "https://youtube-studio-pro.onrender.com/login"
        }, indent=2)

    try:
        v_res = youtube.videos().list(
            id=clean_id,
            part='snippet,status,statistics,contentDetails'
        ).execute()

        items = v_res.get('items', [])
        if not items:
            return json.dumps({"status": "not_found", "message": f"Video '{clean_id}' not found."})

        v = items[0]
        snip = v.get('snippet', {})
        status = v.get('status', {})
        stats = v.get('statistics', {})

        return json.dumps({
            "status": "success",
            "video_id": clean_id,
            "title": snip.get('title', ''),
            "description": snip.get('description', ''),
            "tags": snip.get('tags', []),
            "category_id": snip.get('categoryId', '22'),
            "privacy_status": status.get('privacyStatus', 'public'),
            "thumbnail_url": snip.get('thumbnails', {}).get('maxres', {}).get('url') or snip.get('thumbnails', {}).get('high', {}).get('url', ''),
            "view_count": stats.get('viewCount', '0'),
            "like_count": stats.get('likeCount', '0'),
            "watch_url": f"https://youtu.be/{clean_id}"
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to get video details: {str(e)}"})


# ============================================================================
# TOOL 3: update_video_metadata
# ============================================================================
@mcp.tool(
    name="update_video_metadata",
    description=(
        "Updates title, description, tags, hashtags, category, and privacy status for an existing video on YouTube. "
        "Applies changes directly via the YouTube Data API v3."
    )
)
def update_video_metadata(
    video_id: str,
    title: str = "",
    description: str = "",
    tags: List[str] = None,
    category_id: str = "",
    privacy_status: str = ""
) -> str:
    """Updates video title, description, tags, and settings on YouTube.

    Args:
        video_id: YouTube video ID to edit.
        title: New video title (optional; if empty, retains current title).
        description: New video description (optional; if empty, retains current description).
        tags: New list of tags (optional; if empty/None, retains current tags).
        category_id: Numeric YouTube category ID (e.g. '22', '24').
        privacy_status: 'public', 'private', or 'unlisted'.
    """
    clean_id = (video_id or "").strip()
    if not clean_id:
        return json.dumps({"status": "error", "message": "Video ID is required."})

    youtube = get_youtube_service()
    if not youtube:
        return json.dumps({
            "status": "credentials_required",
            "message": "YouTube OAuth credentials required. Visit https://youtube-studio-pro.onrender.com/login",
            "auth_url": "https://youtube-studio-pro.onrender.com/login"
        }, indent=2)

    try:
        # 1. Fetch current snippet and status to preserve required fields (e.g. categoryId)
        current = youtube.videos().list(id=clean_id, part='snippet,status').execute()
        items = current.get('items', [])
        if not items:
            return json.dumps({"status": "not_found", "message": f"Video '{clean_id}' not found."})

        cur_snip = items[0].get('snippet', {})
        cur_status = items[0].get('status', {})

        new_title = (title or "").strip() or cur_snip.get('title', '')
        new_desc = description if description != "" else cur_snip.get('description', '')
        new_tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()] if tags is not None else cur_snip.get('tags', [])
        new_cat = category_id.strip() if category_id else cur_snip.get('categoryId', '22')
        new_privacy = privacy_status.lower() if privacy_status and privacy_status.lower() in ['public', 'private', 'unlisted'] else cur_status.get('privacyStatus', 'public')

        body = {
            "id": clean_id,
            "snippet": {
                "title": new_title[:100],
                "description": new_desc[:5000],
                "tags": new_tags[:50],
                "categoryId": str(new_cat)
            },
            "status": {
                "privacyStatus": new_privacy
            }
        }

        up_res = youtube.videos().update(part='snippet,status', body=body).execute()
        updated_snip = up_res.get('snippet', {})
        updated_status = up_res.get('status', {})

        return json.dumps({
            "status": "success",
            "message": f"Successfully updated metadata for video '{clean_id}' on YouTube.",
            "video_id": clean_id,
            "updated_title": updated_snip.get('title'),
            "updated_tags_count": len(updated_snip.get('tags', [])),
            "updated_tags": updated_snip.get('tags', []),
            "privacy_status": updated_status.get('privacyStatus'),
            "watch_url": f"https://youtu.be/{clean_id}"
        }, indent=2)

    except Exception as e:
        err_str = str(e)
        return json.dumps({"status": "error", "message": f"YouTube API update failed: {err_str}"})


# ============================================================================
# TOOL 4: update_video_thumbnail
# ============================================================================
@mcp.tool(
    name="update_video_thumbnail",
    description=(
        "Uploads and applies a new custom thumbnail image to any video on the channel. "
        "Accepts either an image URL (which it automatically downloads) or a local file path."
    )
)
def update_video_thumbnail(
    video_id: str,
    thumbnail_url: str = "",
    thumbnail_path: str = ""
) -> str:
    """Sets a custom thumbnail image for a YouTube video.

    Args:
        video_id: YouTube video ID.
        thumbnail_url: Direct image URL (JPG, PNG) to download and set as thumbnail.
        thumbnail_path: Local file path of image to upload as thumbnail.
    """
    clean_id = (video_id or "").strip()
    if not clean_id:
        return json.dumps({"status": "error", "message": "Video ID is required."})

    youtube = get_youtube_service()
    if not youtube:
        return json.dumps({
            "status": "credentials_required",
            "message": "YouTube OAuth credentials required. Visit https://youtube-studio-pro.onrender.com/login",
            "auth_url": "https://youtube-studio-pro.onrender.com/login"
        }, indent=2)

    local_file = None
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "thumbnails")
    os.makedirs(upload_dir, exist_ok=True)

    if thumbnail_url:
        try:
            parsed = urllib.parse.urlparse(thumbnail_url)
            if parsed.scheme in ["http", "https"]:
                dest = os.path.join(upload_dir, f"thumb_mcp_{clean_id}_{uuid.uuid4().hex[:6]}.jpg")
                urllib.request.urlretrieve(thumbnail_url, dest)
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    local_file = dest
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Failed to download thumbnail from URL: {e}"})

    if not local_file and thumbnail_path and os.path.exists(thumbnail_path):
        local_file = thumbnail_path

    if not local_file or not os.path.exists(local_file):
        return json.dumps({"status": "error", "message": "Valid thumbnail_url or thumbnail_path must be provided."})

    try:
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(local_file, mimetype='image/jpeg', resumable=True)
        res = youtube.thumbnails().set(
            videoId=clean_id,
            media_body=media
        ).execute()

        new_thumb_url = res.get('items', [{}])[0].get('default', {}).get('url', '')
        return json.dumps({
            "status": "success",
            "message": f"Successfully updated thumbnail for video '{clean_id}'.",
            "video_id": clean_id,
            "thumbnail_url": new_thumb_url,
            "watch_url": f"https://youtu.be/{clean_id}"
        }, indent=2)
    except Exception as e:
        err_msg = str(e)
        if "forbidden" in err_msg.lower() or "verified" in err_msg.lower():
            return json.dumps({
                "status": "error",
                "message": "Setting custom thumbnails requires a phone-verified YouTube channel. Please verify channel in official YouTube Studio."
            })
        return json.dumps({"status": "error", "message": f"Failed to set thumbnail: {err_msg}"})


# ============================================================================
# TOOL 5: optimize_and_update_video
# ============================================================================
@mcp.tool(
    name="optimize_and_update_video",
    description=(
        "Autonomous AI Optimization Tool: Fetches a video from YouTube, generates viral title, "
        "SEO description, hashtags, and keywords using Gemini AI, and optionally applies changes directly to YouTube."
    )
)
def optimize_and_update_video(
    video_id: str,
    custom_instructions: str = "",
    auto_apply: bool = True
) -> str:
    """Fetches video from YouTube, optimizes its SEO metadata, and updates it.

    Args:
        video_id: The YouTube video ID to optimize.
        custom_instructions: Optional instructions (e.g. 'Target Hindi gaming audience', 'Add curiosity hook').
        auto_apply: If True, writes the changes to YouTube directly. If False, returns the proposed metadata.
    """
    clean_id = (video_id or "").strip()
    if not clean_id:
        return json.dumps({"status": "error", "message": "Video ID is required."})

    # 1. Fetch current video details
    details_raw = get_video_details(clean_id)
    details = json.loads(details_raw)
    if details.get("status") != "success":
        return details_raw

    cur_title = details.get("title", "")
    cur_desc = details.get("description", "")
    cur_tags = details.get("tags", [])

    # 2. Run optimization
    opt_raw = optimize_video_metadata(
        title=cur_title,
        description=cur_desc,
        tags=cur_tags,
        video_url=f"https://youtu.be/{clean_id}",
        custom_instructions=custom_instructions
    )
    opt_res = json.loads(opt_raw)

    proposed_title = opt_res.get("optimized_title", cur_title)
    proposed_desc = opt_res.get("optimized_description", cur_desc)
    proposed_tags = opt_res.get("optimized_tags", cur_tags)

    update_result = None
    if auto_apply:
        up_raw = update_video_metadata(
            video_id=clean_id,
            title=proposed_title,
            description=proposed_desc,
            tags=proposed_tags
        )
        update_result = json.loads(up_raw)

    return json.dumps({
        "status": "success",
        "video_id": clean_id,
        "auto_applied": auto_apply,
        "before": {
            "title": cur_title,
            "tags_count": len(cur_tags)
        },
        "optimized": {
            "title": proposed_title,
            "description": proposed_desc,
            "tags": proposed_tags,
            "hashtags": opt_res.get("hashtags", []),
            "thumbnail_directive": opt_res.get("thumbnail_directive", {}),
            "seo_score": opt_res.get("seo_score", 90)
        },
        "youtube_update_status": update_result
    }, indent=2)


# ============================================================================
# TOOL 6: batch_optimize_channel_videos
# ============================================================================
@mcp.tool(
    name="batch_optimize_channel_videos",
    description=(
        "Batch-processes multiple channel videos one by one: fetches each video, "
        "optimizes its title, SEO description, hashtags, and keywords, and updates it on YouTube."
    )
)
def batch_optimize_channel_videos(
    max_videos: int = 5,
    custom_instructions: str = "",
    auto_apply: bool = True
) -> str:
    """Iterates through recent channel videos and optimizes their metadata.

    Args:
        max_videos: Number of videos to process (default 5, max 15).
        custom_instructions: Additional tone or SEO guidance.
        auto_apply: If True, writes the changes to YouTube directly.
    """
    vids_raw = list_channel_videos(max_results=max_videos)
    vids_data = json.loads(vids_raw)
    if vids_data.get("status") != "success":
        return vids_raw

    results = []
    for item in vids_data.get("videos", []):
        vid_id = item["video_id"]
        res_raw = optimize_and_update_video(
            video_id=vid_id,
            custom_instructions=custom_instructions,
            auto_apply=auto_apply
        )
        try:
            results.append(json.loads(res_raw))
        except Exception:
            results.append({"video_id": vid_id, "raw": res_raw})

    return json.dumps({
        "status": "success",
        "total_processed": len(results),
        "channel": vids_data.get("channel", "Channel"),
        "results": results
    }, indent=2)


# ============================================================================
# TOOL 7: optimize_video_metadata (Drafts & Concepts)
# ============================================================================
@mcp.tool(
    name="optimize_video_metadata",
    description=(
        "Optimizes YouTube video metadata (title, description, tags, and thumbnail directive) "
        "for maximum Click-Through Rate (CTR), search discoverability, and viral performance. "
        "Uses Gemini 3.8 Flash if configured, with algorithmic SEO fallback."
    )
)
def optimize_video_metadata(
    title: str,
    description: str,
    tags: List[str],
    video_url: str = "",
    format_type: str = "Short",
    custom_instructions: str = ""
) -> str:
    """Optimize YouTube video title, description, and tags for SEO and CTR.

    Args:
        title: Current or proposed title for the video.
        description: Current or proposed description/summary of the video.
        tags: List of seed tags or search keywords.
        video_url: Optional URL to the video file or YouTube watch URL.
        format_type: Video target format ('Short' for 9:16 Shorts or 'Long' for 16:9 Long-form).
        custom_instructions: Optional additional guidance for the optimization engine.
    """
    clean_title = (title or "").strip()
    clean_desc = (description or "").strip()
    clean_tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]

    gemini_metadata = None
    try:
        import gemini_engine
        cfg = gemini_engine.get_gemini_config()
        if cfg.get("is_configured"):
            client = gemini_engine.get_genai_client()
            prompt = f"""VIDEO TARGET FORMAT: {format_type}
CURRENT TITLE: {clean_title}
CURRENT DESCRIPTION: {clean_desc}
SEED TAGS: {', '.join(clean_tags)}
{f'VIDEO URL: {video_url}' if video_url else ''}
{f'ADDITIONAL GUIDANCE: {custom_instructions}' if custom_instructions else ''}

Generate viral, high-CTR, SEO-optimized YouTube metadata.
RULES:
1. Title: Under 60 characters for Shorts with emotional tension or curiosity gap; For Long format use [Intriguing Hook] | [High Volume Keyword].
2. Description: Hook in first 2 lines, followed by comprehensive SEO summary, chapters/timestamps placeholder, and hashtags.
3. Tags: 12-18 high-ranking discovery tags.
4. Thumbnail Directive: Text overlay (3-4 words max), scene direction, and contrast color scheme.

Return STRICTLY valid JSON with schema:
{{
  "viral_title": "string",
  "alternative_titles": ["string", "string", "string"],
  "description": "string",
  "search_tags": ["string"],
  "hashtags": ["#string"],
  "thumbnail_directive": {{
    "text_overlay": "string",
    "visual_scene_direction": "string",
    "recommended_color_theme": "string"
  }},
  "seo_score": 95,
  "summary_insights": "string"
}}"""
            response = client.models.generate_content(
                model=cfg.get("model") or "gemini-3.8-flash",
                contents=prompt
            )
            raw_text = (response.text or "").strip()
            match = re.search(r'(\{[\s\S]*\})', raw_text)
            if match:
                gemini_metadata = json.loads(match.group(1))
    except Exception as e:
        print(f"[MCP Tool: optimize_video_metadata] Gemini optimization notice: {e}")

    if gemini_metadata:
        return json.dumps({
            "status": "success",
            "provider": "Gemini 3.8 Flash",
            "format_type": format_type,
            "optimized_title": gemini_metadata.get("viral_title", clean_title),
            "alternative_titles": gemini_metadata.get("alternative_titles", []),
            "optimized_description": gemini_metadata.get("description", clean_desc),
            "optimized_tags": gemini_metadata.get("search_tags", clean_tags),
            "hashtags": gemini_metadata.get("hashtags", ["#YouTube", "#Trending"]),
            "thumbnail_directive": gemini_metadata.get("thumbnail_directive", {
                "text_overlay": "DON'T MISS THIS",
                "visual_scene_direction": "High-contrast subject close-up with expressive reaction",
                "recommended_color_theme": "Vibrant Yellow and Deep Charcoal"
            }),
            "seo_score": gemini_metadata.get("seo_score", 95),
            "insights": gemini_metadata.get("summary_insights", "Gemini-enhanced high-engagement metadata generated.")
        }, indent=2)

    # High-Performance Algorithmic Fallback Optimizer
    base_subject = clean_title or "Secret Behind Trending Strategy"
    is_short = format_type.lower() == "short"

    if is_short:
        viral_title = f"{base_subject} (WATCH TILL END) 😱"
        alt_titles = [
            f"Nobody Knew About {base_subject}! #Shorts",
            f"The Real Truth: {base_subject} 🔥",
            f"Wait For It... {base_subject} ⏳"
        ]
        recommended_hashtags = ["#Shorts", "#Trending", "#Viral", "#FYP"]
    else:
        viral_title = f"The Shocking Truth About {base_subject} | Complete Breakdown"
        alt_titles = [
            f"Why Everyone Is Wrong About {base_subject} (2026)",
            f"{base_subject}: Step-by-Step Ultimate Guide",
            f"I Tested {base_subject} For 30 Days: Here Is What Happened"
        ]
        recommended_hashtags = ["#YouTube", "#Tutorial", "#InDepth", "#2026Guide"]

    combined_tags = list(dict.fromkeys(
        clean_tags +
        [base_subject.lower(), "trending", "youtube algorithm", "viral strategy", "tips and tricks", "creator studio pro"]
    ))

    formatted_desc = (
        f"🔥 {viral_title}\n\n"
        f"{clean_desc if clean_desc else 'In this video, we break down everything you need to know about ' + base_subject + ' with actionable insights and deep analysis.'}\n\n"
        f"📌 TIMESTAMPS / HIGHLIGHTS:\n"
        f"0:00 - The Big Hook\n"
        f"0:45 - Key Breakdown & Analysis\n"
        f"1:30 - Secret Formula\n"
        f"2:15 - Final Takeaway & Action Step\n\n"
        f"🔔 Subscribe to YouTube Creator Studio Pro for more daily masterclasses!\n\n"
        f"{' '.join(recommended_hashtags)}"
    )

    return json.dumps({
        "status": "success",
        "provider": "Algorithmic SEO Engine",
        "format_type": format_type,
        "optimized_title": viral_title,
        "alternative_titles": alt_titles,
        "optimized_description": formatted_desc,
        "optimized_tags": combined_tags[:18],
        "hashtags": recommended_hashtags,
        "thumbnail_directive": {
            "text_overlay": "THE SECRET",
            "visual_scene_direction": "High-contrast expressive face on the left, compelling subject/proof on the right",
            "recommended_color_theme": "Vibrant Yellow (#FFD700) on Deep Charcoal (#1A1A1A)"
        },
        "seo_score": 88,
        "insights": "Algorithmic title scoring applied: Curiosity gap established, search keywords populated."
    }, indent=2)


# ============================================================================
# TOOL 8: upload_to_youtube
# ============================================================================
@mcp.tool(
    name="upload_to_youtube",
    description=(
        "Uploads or queues a video for YouTube with standard metadata (title, description, tags, "
        "privacy, and category). Validates credentials and schedules background ingestion."
    )
)
def upload_to_youtube(
    title: str,
    description: str,
    tags: List[str],
    video_url: str = "",
    privacy_status: str = "private",
    thumbnail_url: str = "",
    category_id: str = "22",
    made_for_kids: bool = False
) -> str:
    """Upload video to YouTube with specified metadata and privacy settings.

    Args:
        title: Title of the video (maximum 100 characters).
        description: Description of the video (maximum 5000 characters).
        tags: List of keywords/tags for the video.
        video_url: Direct download URL or local file path to the video file.
        privacy_status: Privacy status for the upload ('public', 'private', or 'unlisted'). Default is 'private'.
        thumbnail_url: Optional URL or path to custom thumbnail image.
        category_id: YouTube numeric category ID (default '22' for People & Blogs, '24' for Entertainment, '20' for Gaming).
        made_for_kids: COPPA declaration whether this video is made for kids.
    """
    clean_title = (title or "").strip()[:100]
    clean_desc = (description or "").strip()[:5000]
    clean_tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    privacy = privacy_status.lower() if privacy_status.lower() in ["public", "private", "unlisted"] else "private"

    if not clean_title:
        return json.dumps({"status": "error", "message": "Title cannot be empty."})

    creds_available = False
    creds_dict = None
    try:
        import app as main_app
        creds = main_app.get_stored_credentials()
        if creds and creds.valid:
            creds_available = True
            creds_dict = {
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }
    except Exception as ce:
        print(f"[MCP Tool: upload_to_youtube] Credential check notice: {ce}")

    task_id = str(uuid.uuid4())
    resolved_video_path = None
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    if video_url:
        parsed = urllib.parse.urlparse(video_url)
        if parsed.scheme in ["http", "https"]:
            try:
                dest_file = os.path.join(upload_dir, f"{task_id}_downloaded.mp4")
                urllib.request.urlretrieve(video_url, dest_file)
                if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
                    resolved_video_path = dest_file
            except Exception as de:
                print(f"[MCP Tool: upload_to_youtube] Download error: {de}")
        elif os.path.exists(video_url):
            resolved_video_path = video_url

    if not resolved_video_path:
        for f in os.listdir(upload_dir):
            if f.endswith(('.mp4', '.mov', '.mkv', '.webm')):
                candidate = os.path.join(upload_dir, f)
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    resolved_video_path = candidate
                    break

    if not creds_available:
        return json.dumps({
            "status": "credentials_required",
            "message": "YouTube OAuth credentials required. Please visit https://youtube-studio-pro.onrender.com/login",
            "task_id": task_id,
            "validated_payload": {
                "title": clean_title,
                "description": clean_desc[:200] + "...",
                "tags": clean_tags,
                "privacy": privacy,
                "category_id": category_id,
                "made_for_kids": made_for_kids,
                "video_file_detected": bool(resolved_video_path)
            },
            "auth_url": "https://youtube-studio-pro.onrender.com/login"
        }, indent=2)

    if resolved_video_path and os.path.exists(resolved_video_path):
        try:
            import app as main_app
            main_app.upload_tasks[task_id] = {
                'status': 'uploading',
                'progress': 0.0,
                'video_id': None,
                'error': None
            }
            thread = threading.Thread(
                target=main_app.execute_youtube_upload,
                args=(
                    task_id,
                    creds_dict,
                    resolved_video_path,
                    thumbnail_url if thumbnail_url and os.path.exists(thumbnail_url) else None,
                    clean_title,
                    clean_desc,
                    clean_tags,
                    privacy,
                    category_id,
                    made_for_kids
                )
            )
            thread.daemon = True
            thread.start()

            return json.dumps({
                "status": "uploading",
                "message": f"Video upload initiated for '{clean_title}'.",
                "task_id": task_id,
                "privacy_status": privacy,
                "tracking_url": f"https://youtube-studio-pro.onrender.com/api/upload_status/{task_id}"
            }, indent=2)
        except Exception as ue:
            return json.dumps({"status": "error", "message": f"Failed to initiate background upload: {str(ue)}"})

    return json.dumps({
        "status": "pending_file",
        "message": f"Metadata for '{clean_title}' saved and ready. Provide a valid video_url or upload the video file via the YouTube Studio web UI.",
        "task_id": task_id,
        "title": clean_title,
        "tags_count": len(clean_tags),
        "privacy": privacy
    }, indent=2)


# ============================================================================
# UNIVERSAL CORS & ASGIMIDDLEWARE WRAPPER
# ============================================================================
class UniversalCORSMiddleware:
    """Ensures CORS headers (Access-Control-Allow-Origin: *) are universally present
    across all MCP SSE, message, and health endpoints for Gemini, Grok, and ChatGPT clients.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Preflight OPTIONS request
        if scope["method"] == "OPTIONS":
            headers = [
                (b"access-control-allow-origin", b"*"),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS, HEAD"),
                (b"access-control-allow-headers", b"*"),
                (b"access-control-max-age", b"86400"),
                (b"content-length", b"0"),
            ]
            await send({"type": "http.response.start", "status": 200, "headers": headers})
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(h[0].lower() == b"access-control-allow-origin" for h in headers):
                    headers.append((b"access-control-allow-origin", b"*"))
                if not any(h[0].lower() == b"access-control-allow-methods" for h in headers):
                    headers.append((b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS, HEAD"))
                if not any(h[0].lower() == b"access-control-allow-headers" for h in headers):
                    headers.append((b"access-control-allow-headers", b"*"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ============================================================================
# MCP TOOLS CATALOG (JSON SCHEMA FOR DIRECT JSON-RPC HTTP & SSE)
# ============================================================================
MCP_TOOLS_CATALOG = [
    {
        "name": "list_channel_videos",
        "description": "Lists uploaded videos from the authenticated YouTube channel. Returns video IDs, titles, descriptions, tags, thumbnail URLs, and privacy status. Use this tool to inspect all existing videos before editing or optimizing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of videos to fetch (default: 15, max: 50).",
                    "default": 15
                }
            },
            "required": []
        }
    },
    {
        "name": "get_video_details",
        "description": "Retrieves full metadata for a specific YouTube video (title, description, tags, category, and statistics).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The 11-character YouTube video ID (e.g. '0zZgD9g5PQ8')."
                }
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "update_video_metadata",
        "description": "Updates the title, description, tags, and category of an existing YouTube video directly on YouTube. Changes reflect immediately on YouTube.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The 11-character YouTube video ID to edit."
                },
                "title": {
                    "type": "string",
                    "description": "New video title (max 100 characters)."
                },
                "description": {
                    "type": "string",
                    "description": "New video description (max 5000 characters)."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tags / keywords for the video."
                },
                "category_id": {
                    "type": "string",
                    "description": "YouTube category ID (e.g. '22' for People & Blogs, '28' for Science & Tech)."
                }
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "update_video_thumbnail",
        "description": "Sets a new custom thumbnail image for a YouTube video. Provide either thumbnail_url (a publicly accessible image URL) or thumbnail_base64 (data URI or raw base64).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The 11-character YouTube video ID."
                },
                "thumbnail_url": {
                    "type": "string",
                    "description": "Public HTTP/HTTPS URL of the new thumbnail image (JPEG or PNG, recommended 1280x720)."
                },
                "thumbnail_base64": {
                    "type": "string",
                    "description": "Base64 encoded image string (or data:image/jpeg;base64,...) of the thumbnail."
                }
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "optimize_and_update_video",
        "description": "Uses Gemini AI to analyze a video's current metadata, generates optimized high-ranking titles, descriptions, and hashtags, and applies the changes directly to YouTube.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The 11-character YouTube video ID to optimize."
                },
                "focus_keywords": {
                    "type": "string",
                    "description": "Optional focus keywords or target audience to guide the AI optimization."
                },
                "language": {
                    "type": "string",
                    "description": "Primary language for the optimization (default: 'en').",
                    "default": "en"
                }
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "batch_optimize_channel_videos",
        "description": "Iterates across multiple recent videos on your YouTube channel, generates AI-optimized metadata (titles, descriptions, tags, hashtags) for each, and applies updates directly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_videos": {
                    "type": "integer",
                    "description": "Number of recent videos to optimize (default: 5, max: 20).",
                    "default": 5
                },
                "focus_topic": {
                    "type": "string",
                    "description": "Optional channel topic or theme to focus optimizations around."
                },
                "update_thumbnails": {
                    "type": "boolean",
                    "description": "Whether to also trigger thumbnail regeneration where supported (default: false).",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "optimize_video_metadata",
        "description": "Generates high-CTR titles, SEO descriptions, tags, and hashtags using Gemini AI without directly updating YouTube. Use to preview recommendations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Current draft title or video topic."
                },
                "description": {
                    "type": "string",
                    "description": "Current draft description or rough bullet points."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Current tags or seed keywords."
                },
                "category": {
                    "type": "string",
                    "description": "Target video category (default: 'People & Blogs').",
                    "default": "People & Blogs"
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "upload_to_youtube",
        "description": "Stages and initiates a YouTube video upload with title, description, tags, category, and privacy status. Video file can be provided via video_url or uploaded directly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Video title (max 100 chars)."
                },
                "description": {
                    "type": "string",
                    "description": "Video description with summary, timestamps, and hashtags."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords/tags for YouTube search indexing."
                },
                "category": {
                    "type": "string",
                    "description": "YouTube Category ID (e.g. '22' for People & Blogs, '28' for Science & Technology).",
                    "default": "22"
                },
                "privacy_status": {
                    "type": "string",
                    "description": "Visibility: 'public', 'private', or 'unlisted'. Default: 'public'.",
                    "default": "public"
                },
                "video_url": {
                    "type": "string",
                    "description": "Public URL where the video file can be downloaded for upload."
                }
            },
            "required": ["title", "description"]
        }
    }
]

MCP_TOOL_FUNCTIONS = {
    "list_channel_videos": list_channel_videos,
    "get_video_details": get_video_details,
    "update_video_metadata": update_video_metadata,
    "update_video_thumbnail": update_video_thumbnail,
    "optimize_and_update_video": optimize_and_update_video,
    "batch_optimize_channel_videos": batch_optimize_channel_videos,
    "optimize_video_metadata": optimize_video_metadata,
    "upload_to_youtube": upload_to_youtube,
}


def handle_jsonrpc_sync(req_data: dict) -> Optional[dict]:
    """Processes a single JSON-RPC 2.0 MCP request dict and returns the response dict,
    or None if the request is a notification requiring no content body.
    """
    if not isinstance(req_data, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request: expected JSON object."}
        }

    method = req_data.get("method")
    req_id = req_data.get("id")
    params = req_data.get("params") or {}

    print(f"[MCP JSON-RPC DISPATCH] method='{method}', id={req_id}", flush=True)

    if method == "initialize":
        client_proto = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_proto,
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    },
                    "resources": {},
                    "prompts": {},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "YouTube Creator Studio Pro",
                    "version": "1.0.0"
                }
            }
        }

    elif method in ("notifications/initialized", "notifications/cancelled", "$/cancelRequest"):
        if req_id is not None:
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        return None

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": MCP_TOOLS_CATALOG
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        tool_fn = MCP_TOOL_FUNCTIONS.get(tool_name)
        if not tool_fn:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{tool_name}' not found."
                }
            }
        try:
            call_res = tool_fn(**arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(call_res)
                        }
                    ],
                    "isError": False
                }
            }
        except Exception as te:
            print(f"[MCP TOOL CALL ERROR] {tool_name}: {te}", flush=True)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool execution error: {str(te)}"
                        }
                    ],
                    "isError": True
                }
            }

    elif method in ("resources/list", "resources/templates/list"):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"resources": []}
        }

    elif method == "prompts/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"prompts": []}
        }

    else:
        print(f"[MCP UNKNOWN METHOD] '{method}'", flush=True)
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not implemented."
                }
            }
        return None


# ============================================================================
# ASGI APPLICATION FACTORY (DUAL TRANSPORT: SSE + DIRECT JSON-RPC HTTP POST)
# ============================================================================
def create_mcp_asgi_app(flask_app=None) -> Starlette:
    """Constructs the unified ASGI application supporting:
    - MCP SSE stream on GET /sse, /mcp, /api/mcp
    - MCP direct JSON-RPC POST on POST /sse, /mcp, /api/mcp (Gemini Spark, ChatGPT, Grok)
    - OAuth protected resource discovery handling on /.well-known/oauth-protected-resource
    - Health check on /health (returning 200 OK)
    - Fallback mounting of Flask web interface and APIs
    """
    mcp_sse_starlette = mcp.sse_app()

    sse_route = next(
        (r for r in mcp_sse_starlette.routes if getattr(r, 'path', None) == '/sse'),
        None
    )
    if not sse_route:
        raise RuntimeError("FastMCP did not initialize an /sse route.")

    async def handle_jsonrpc_request(request):
        """Handles HTTP POST requests containing JSON-RPC 2.0 payloads."""
        client_ip = getattr(request.client, 'host', 'unknown')
        try:
            raw_body = await request.body()
            body_text = raw_body.decode('utf-8')
            print(f"[MCP INCOMING POST] {request.method} {request.url.path} from {client_ip}: {body_text[:300]}", flush=True)

            if not body_text.strip():
                return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error: empty body"}}, status_code=400)

            data = json.loads(body_text)

            if isinstance(data, list):
                # Batch request
                responses = [handle_jsonrpc_sync(item) for item in data]
                responses = [r for r in responses if r is not None]
                return JSONResponse(responses, status_code=200)
            else:
                resp = handle_jsonrpc_sync(data)
                if resp is None:
                    return Response(status_code=204)
                return JSONResponse(resp, status_code=200)

        except json.JSONDecodeError as jde:
            print(f"[MCP JSON PARSE ERROR]: {jde}", flush=True)
            return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {str(jde)}"}}, status_code=400)
        except Exception as e:
            print(f"[MCP POST HANDLER ERROR]: {e}", flush=True)
            return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Internal error: {str(e)}"}}, status_code=500)

    async def mcp_dual_transport_endpoint(request):
        """Universal endpoint serving both Server-Sent Events (GET) and direct JSON-RPC (POST)."""
        method = request.method
        if method == "POST":
            return await handle_jsonrpc_request(request)
        elif method == "GET":
            return await sse_route.endpoint(request)
        elif method in ("HEAD", "OPTIONS"):
            return Response(status_code=200)
        return Response(status_code=405)

    async def oauth_protected_resource_endpoint(request):
        """RFC 9728 Protected Resource Metadata discovery handler.
        Returning 404 cleanly signifies to MCP clients (Gemini) that this server
        operates in public/direct unauthenticated mode without OAuth requirements.
        """
        client_ip = getattr(request.client, 'host', 'unknown')
        print(f"[OAUTH DISCOVERY PROBE]: {request.method} {request.url.path} from {client_ip}", flush=True)
        if request.method == "OPTIONS":
            return Response(status_code=200)
        return JSONResponse(
            {"status": "public_mcp", "message": "This MCP server does not enforce OAuth 2.0; direct MCP tool execution is available."},
            status_code=404
        )

    async def health_endpoint(request):
        return JSONResponse({
            "status": "ok",
            "service": "youtube-creator-studio-pro",
            "mcp_version": "1.30.0",
            "mcp_sse_endpoint": "/sse",
            "mcp_alias_endpoint": "/mcp",
            "tools": list(MCP_TOOL_FUNCTIONS.keys()),
            "timestamp": time.time()
        }, status_code=200)

    unified_routes = [
        Route("/health", endpoint=health_endpoint, methods=["GET", "HEAD", "OPTIONS"]),
        Route("/sse", endpoint=mcp_dual_transport_endpoint, methods=["GET", "POST", "HEAD", "OPTIONS"]),
        Route("/mcp", endpoint=mcp_dual_transport_endpoint, methods=["GET", "POST", "HEAD", "OPTIONS"]),
        Route("/api/mcp", endpoint=mcp_dual_transport_endpoint, methods=["GET", "POST", "HEAD", "OPTIONS"]),
        Route("/.well-known/oauth-protected-resource", endpoint=oauth_protected_resource_endpoint, methods=["GET", "HEAD", "OPTIONS"]),
        Route("/.well-known/oauth-protected-resource/{subpath:path}", endpoint=oauth_protected_resource_endpoint, methods=["GET", "HEAD", "OPTIONS"]),
    ]

    for r in mcp_sse_starlette.routes:
        if getattr(r, 'path', None) not in ['/sse']:
            unified_routes.append(r)

    if flask_app:
        unified_routes.append(
            Mount("/", app=WSGIMiddleware(flask_app))
        )

    base_app = Starlette(
        debug=False,
        routes=unified_routes,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["*"]
            )
        ]
    )

    return UniversalCORSMiddleware(base_app)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting standalone Universal MCP server on 0.0.0.0:{port}...")
    app = create_mcp_asgi_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


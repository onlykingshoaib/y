import json
import sys
import os
import re
from google import genai
from google.genai import types

def run_direct_pipeline(video_url: str, api_key: str):
    """
    Direct, lightweight, standalone YouTube Video Optimizer.
    Passes YouTube URL directly into Gemini native vision.
    Saves clean, factual, production-ready JSON output.
    """
    if not api_key:
        print("[Error] Gemini API Key is required.")
        sys.exit(1)

    print(f"Connecting to Gemini API...")
    client = genai.Client(api_key=api_key)

    # 1. सीधे वीडियो लिंक को नेटिव विज़न में भेजना
    video_part = types.Part(
        file_data=types.FileData(file_uri=video_url.strip())
    )

    prompt = """
    Analyze this YouTube video completely and return ONLY a valid JSON object.
    Do not use generic text, filler lines, or placeholders.
    Extract the exact narrative, figures, or topic shown.

    JSON Schema:
    {
      "title": "High-impact title strictly under 65 characters",
      "category_id": "Exact YouTube category numeric ID",
      "description": "Factual 2-3 sentence summary of the exact events followed by 3 hashtags",
      "tags": ["10-15 hyper-relevant niche keywords"],
      "poster_prompt": "Cinematic visual prompt for a 4K thumbnail"
    }
    """

    print(f"Analyzing video via Native Vision ({video_url})...")
    
    # Try gemini-3.8-flash first, fallback to gemini-2.5-flash if unavailable
    candidate_models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    response = None
    used_model = None

    for model_name in candidate_models:
        try:
            print(f"Sending request to model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=[video_part, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            if response and response.text:
                used_model = model_name
                break
        except Exception as e:
            print(f"Notice: Model {model_name} returned: {e}")

    if not response or not response.text:
        print("[Error] Failed to generate metadata from the video.")
        sys.exit(1)

    # Clean code blocks if present
    raw_text = response.text.strip()
    clean_json = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
    clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE)
    clean_json = re.sub(r"```$", "", clean_json.strip())

    try:
        data = json.loads(clean_json)
    except json.JSONDecodeError as jde:
        print(f"[Error] Failed to parse JSON response: {jde}")
        print("Raw output:", raw_text)
        sys.exit(1)

    # Enforce title under 65 characters
    if "title" in data and len(data["title"]) > 65:
        truncated = data["title"][:65]
        if " " in truncated:
            data["title"] = truncated.rsplit(" ", 1)[0].strip()
        else:
            data["title"] = truncated.strip()

    data["engine"] = used_model

    # 2. फ़ाइनल रिज़ल्ट को साफ़ फ़ाइल में सेव करना
    output_filename = "final_optimized_metadata.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[Success] काम पूरा हुआ। सारा डेटा '{output_filename}' में सुरक्षित लिख दिया गया है:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data

if __name__ == "__main__":
    # उपयोग: python direct_optimizer.py "YOUTUBE_URL" "YOUR_GEMINI_API_KEY"
    if len(sys.argv) < 3:
        # Check if environment variable or saved config has API key
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if len(sys.argv) == 2 and env_key:
            run_direct_pipeline(sys.argv[1], env_key)
        else:
            print("Usage: python direct_optimizer.py <YOUTUBE_URL> <GEMINI_API_KEY>")
            sys.exit(1)
    else:
        run_direct_pipeline(sys.argv[1], sys.argv[2])

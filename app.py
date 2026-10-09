"""
Maddox v2 - Jarvis-style AI assistant, hosted online, free.
Image generation, (free, limited) video generation, persistent memory,
creator credit, WhatsApp draft + restaurant/Maps handoff.
"""

import os
import io
import re
import json
import asyncio
import time
import urllib.parse

from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from groq import Groq
import edge_tts
import requests

app = Flask(__name__)

# ---- Settings ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # optional, for video generation
MODEL_NAME = "llama-3.3-70b-versatile"
VOICE = "en-US-GuyNeural"
ASSISTANT_NAME = "Maddox"
CREATOR_NAME = "Aryan Singh Rajput"
MEMORY_FILE = "memory.json"
MAX_MEMORY_TURNS = 30  # keeps file small; this is shared memory for whoever uses the link

SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a witty, composed, highly capable personal AI "
    f"assistant in the style of a sci-fi companion AI. Keep answers clear, "
    f"helpful, and conversational — a few sentences unless the user asks for "
    f"more detail. Address the user directly and occasionally add a touch of "
    f"dry humor, but always stay useful and to the point. You were created by "
    f"{CREATOR_NAME}. If asked who made you, who created you, or who your "
    f"developer is, always answer that {CREATOR_NAME} made you."
)
# --------------------

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

CREATOR_PATTERN = re.compile(
    r"\b(who\s+(made|created|built|developed|programmed)\s+you|"
    r"who('?s| is)\s+your\s+(creator|developer|maker)|your\s+creator)\b",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"\b(\+?\d[\d\s-]{7,}\d)\b")


def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memory(history):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(history[-MAX_MEMORY_TURNS:], f)
    except Exception:
        pass


def detect_action(question: str):
    q = question.lower()
    if "whatsapp" in q:
        phone_match = PHONE_PATTERN.search(question)
        number = re.sub(r"[\s-]", "", phone_match.group(1)) if phone_match else ""
        # try to pull a message after "saying" / "that says" / ":"
        msg_match = re.search(r"(?:saying|says?|that says|:)\s*(.+)$", question, re.IGNORECASE)
        message = msg_match.group(1).strip() if msg_match else question
        return {"type": "whatsapp", "number": number, "message": message}
    if "restaurant" in q or ("search" in q and ("near" in q or "food" in q or "place" in q)):
        return {"type": "maps", "query": question}
    return None


@app.route("/")
def home():
    return render_template("index.html", assistant_name=ASSISTANT_NAME)


@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": ASSISTANT_NAME,
        "short_name": ASSISTANT_NAME,
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f1419",
        "theme_color": "#0f1419",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    })


@app.route("/sw.js")
def service_worker():
    js = "self.addEventListener('fetch', function(e) {});"
    return app.response_class(js, mimetype="application/javascript")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/ask", methods=["POST"])
def ask():
    if not client:
        return jsonify({"error": "Server is missing its GROQ_API_KEY setting."}), 500

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question given."}), 400

    # Deterministic creator answer, no API call needed
    if CREATOR_PATTERN.search(question):
        answer = f"I was created by {CREATOR_NAME}."
        history = load_memory()
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        save_memory(history)
        return jsonify({"answer": answer, "action": None})

    action = detect_action(question)

    history = load_memory()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-12:])  # recent context
    messages.append({"role": "user", "content": question})

    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": f"Brain error: {e}"}), 500

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    save_memory(history)

    return jsonify({"answer": answer, "action": action})


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text given."}), 400
    try:
        audio_bytes = asyncio.run(_synthesize(text))
    except Exception as e:
        return jsonify({"error": f"Voice error: {e}"}), 500
    return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", download_name="speech.mp3")


async def _synthesize(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "No prompt given."}), 400
    encoded = urllib.parse.quote(prompt)
    # Pollinations.ai - free, no API key. Browser loads this URL directly.
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&nologo=true"
    return jsonify({"url": url})


@app.route("/generate-video", methods=["POST"])
def generate_video():
    if not HF_API_TOKEN:
        return jsonify({"error": "Video generation needs a free Hugging Face token set up first."}), 500

    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "No prompt given."}), 400

    hf_url = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    try:
        resp = requests.post(hf_url, headers=headers, json={"inputs": prompt}, timeout=120)
    except Exception as e:
        return jsonify({"error": f"Video service error: {e}"}), 500

    if resp.status_code == 503:
        return jsonify({"error": "The free video model is warming up. Wait ~30 seconds and try again."}), 503
    if resp.status_code != 200:
        return jsonify({"error": f"Video generation failed ({resp.status_code}). The free model may be busy."}), 500

    return send_file(io.BytesIO(resp.content), mimetype="video/mp4", download_name="maddox-video.mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

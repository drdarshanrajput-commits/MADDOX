"""RIFT - personal AI assistant web application."""

import os
import io
import re
import json
import asyncio
import urllib.parse
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from groq import Groq
import edge_tts

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MEMORY_FILE = BASE_DIR / "memory.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB JSON request limit

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"
VOICE = "en-US-GuyNeural"
ASSISTANT_NAME = "RIFT"
CREATOR_NAME = "Aryan Singh Rajput"
MAX_MEMORY_TURNS = 30

SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a capable, composed, friendly personal AI assistant. "
    "Give accurate, useful answers. Explain difficult topics clearly, distinguish facts "
    "from uncertainty, and never pretend to have performed an action you did not perform. "
    "You can help with ideas, writing, learning, planning, and technical problem-solving. "
    f"Your creator is {CREATOR_NAME}. If asked who Aryan Singh Rajput is, or who created "
    f"you, explain that {CREATOR_NAME} is the creator of RIFT. Do not invent claims about "
    "his public achievements or fame."
)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

CREATOR_PATTERN = re.compile(
    r"\b(?:who\s+(?:made|created|built|developed|programmed)\s+(?:you|rift)|"
    r"who\s+(?:is|was)\s+(?:aryan(?:\s+singh)?(?:\s+rajput)?|your\s+creator)|"
    r"who(?:'s|\s+is)\s+your\s+(?:creator|developer|maker)|"
    r"your\s+creator)\b",
    re.IGNORECASE,
)
IMAGE_INTENT_PATTERN = re.compile(
    r"\b(?:generate|create|make|draw|paint|design)\s+(?:me\s+)?"
    r"(?:an?\s+)?(?:image|picture|photo|illustration|artwork|drawing)\s+"
    r"(?:of\s+)?(.+)$",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"\b(\+?\d[\d\s-]{7,}\d)\b")


def load_memory():
    try:
        if MEMORY_FILE.exists():
            with MEMORY_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_memory(history):
    try:
        with MEMORY_FILE.open("w", encoding="utf-8") as handle:
            json.dump(history[-MAX_MEMORY_TURNS:], handle, ensure_ascii=False)
    except OSError:
        app.logger.exception("Could not save conversation memory")


def detect_action(question: str):
    q = question.lower()
    if "whatsapp" in q:
        phone_match = PHONE_PATTERN.search(question)
        number = re.sub(r"[\s-]", "", phone_match.group(1)) if phone_match else ""
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
        "name": "RIFT — Personal AI",
        "short_name": "RIFT",
        "description": "Your personal AI system",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#050b16",
        "theme_color": "#071426",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    })


@app.route("/sw.js")
def service_worker():
    js = "self.addEventListener('fetch', function(event) {});"
    response = app.response_class(js, mimetype="application/javascript")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(STATIC_DIR, "favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a message."}), 400
    if len(question) > 12000:
        return jsonify({"error": "That message is too long. Please shorten it."}), 413

    image_match = IMAGE_INTENT_PATTERN.search(question)
    if image_match and image_match.group(1).strip():
        prompt = image_match.group(1).strip().rstrip(".!?")
        return jsonify({
            "answer": "Preparing your image now. The image service may take a little time to respond.",
            "action": {"type": "generate_image", "prompt": prompt},
        })

    if CREATOR_PATTERN.search(question):
        answer = (
            f"Aryan Singh Rajput is the creator behind my circuits — the person who "
            f"imagined and developed RIFT. Without him, there would be no RIFT."
        )
        history = load_memory()
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        save_memory(history)
        return jsonify({"answer": answer, "action": None})

    if not client:
        return jsonify({"error": "The server is missing its GROQ_API_KEY setting."}), 500

    history = load_memory()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": question})

    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            answer = "I couldn't form a response just now. Please try again."
    except Exception:
        app.logger.exception("Groq request failed")
        return jsonify({"error": "RIFT couldn't reach its AI model. Please try again shortly."}), 502

    history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    save_memory(history)
    return jsonify({"answer": answer, "action": detect_action(question)})


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text given."}), 400
    if len(text) > 5000:
        return jsonify({"error": "The speech text is too long."}), 413
    try:
        audio_bytes = asyncio.run(_synthesize(text))
    except Exception:
        app.logger.exception("Text-to-speech failed")
        return jsonify({"error": "Voice service is temporarily unavailable."}), 502
    return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", download_name="rift-speech.mp3")


async def _synthesize(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, VOICE)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    return buffer.getvalue()


@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Please describe the image you want."}), 400
    if len(prompt) > 1500:
        return jsonify({"error": "Please shorten the image description."}), 413
    encoded = urllib.parse.quote(prompt)
    # Pollinations is an external service; availability and terms are controlled by its provider.
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
    return jsonify({"url": url})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

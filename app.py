"""
Maddox - Jarvis-style AI assistant, hosted online, free.
Works from any device's browser: PC, phone, smart TV.

Brain: Groq (free API)
Voice: edge-tts (free, natural-sounding)
"""

import os
import io
import asyncio

from flask import Flask, request, jsonify, send_file, render_template
from groq import Groq
import edge_tts

app = Flask(__name__)

# ---- Settings ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
VOICE = "en-US-GuyNeural"  # a calm, confident male voice; see README for other options
ASSISTANT_NAME = "Maddox"

SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a witty, composed, highly capable personal AI "
    f"assistant in the style of a sci-fi companion AI. Keep answers clear, "
    f"helpful, and conversational — a few sentences unless the user asks for "
    f"more detail. Address the user directly and occasionally add a touch of "
    f"dry humor, but always stay useful and to the point."
)
# --------------------

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@app.route("/")
def home():
    return render_template("index.html", assistant_name=ASSISTANT_NAME)


@app.route("/ask", methods=["POST"])
def ask():
    if not client:
        return jsonify({"error": "Server is missing its GROQ_API_KEY setting."}), 500

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question given."}), 400

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": f"Brain error: {e}"}), 500

    return jsonify({"answer": answer})


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

    return send_file(
        io.BytesIO(audio_bytes),
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name="speech.mp3",
    )


async def _synthesize(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

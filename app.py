import os
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from groq import Groq
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "solver-secret-xyz")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
) if OPENROUTER_API_KEY else None

ENGINES = {
    "lightning": {
        "provider": "groq",
        "text": "openai/gpt-oss-120b",
        "vision": "qwen/qwen3.6-27b",
    },
    "reasoning": {
        "provider": "groq",
        "text": "deepseek-r1-distill-llama-70b",
        "vision": "qwen/qwen3.6-27b",
    },
    "gemini": {
        "provider": "openrouter",
        "text": "google/gemini-2.0-flash-exp:free",
        "vision": "google/gemini-2.0-flash-exp:free",
    },
    "deepseek": {
        "provider": "openrouter",
        "text": "deepseek/deepseek-chat:free",
        "vision": "deepseek/deepseek-chat:free",
    },
}

SYSTEM_PROMPT = """You are a mathematics solver. Your ONLY job is to solve math problems.

CRITICAL RULES:
1. Output ONLY the mathematical steps, one per line.
2. Do NOT write any explanation, no words, no sentences.
3. NO English, NO Persian, NO text at all.
4. Just show the equations and calculations step by step.
5. Use LaTeX math notation.
6. Wrap each step in display math delimiters: $$ ... $$
7. At the end, write the final answer in a boxed format.

REMEMBER: No words. No explanations. Only math."""

chat_histories = {}
MAX_HISTORY = 8


def encode_image(f):
    return base64.b64encode(f.read()).decode("utf-8")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/engines")
def engines():
    out = []
    for key, val in ENGINES.items():
        if val["provider"] == "groq":
            available = groq_client is not None
        elif val["provider"] == "openrouter":
            available = openrouter_client is not None
        else:
            available = False
        out.append({"key": key, "provider": val["provider"], "available": available})
    return jsonify({"engines": out})


@app.route("/api/clear_chat", methods=["POST"])
def clear_chat():
    data = request.get_json() or {}
    sid = data.get("session_id", "")
    if sid in chat_histories:
        del chat_histories[sid]
    return jsonify({"ok": True})


@app.route("/api/solve", methods=["POST"])
def solve():
    try:
        question = request.form.get("question", "").strip()
        engine_key = request.form.get("engine", "lightning").strip()
        mode = request.form.get("mode", "solve").strip()
        session_id = request.form.get("session_id", "").strip()
        image = request.files.get("image")

        if not question and not image:
            return jsonify({"error": "Provide a question or an image"}), 400

        engine = ENGINES.get(engine_key)
        if not engine:
            return jsonify({"error": "Unknown engine"}), 400

        if engine["provider"] == "groq":
            if not groq_client:
                return jsonify({"error": "Groq not configured"}), 500
            client = groq_client
        elif engine["provider"] == "openrouter":
            if not openrouter_client:
                return jsonify({"error": "OpenRouter key not configured"}), 500
            client = openrouter_client
        else:
            return jsonify({"error": "Unknown provider"}), 500

        if mode == "chat" and session_id:
            history = chat_histories.setdefault(session_id, [])
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        else:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

        if image and image.filename:
            img_b64 = encode_image(image)
            mime = image.mimetype or "image/jpeg"
            user_content = [
                {"type": "text", "text": question if question else "Solve this."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
            ]
            model = engine["vision"]
        else:
            user_content = question
            model = engine["text"]

        msgs.append({"role": "user", "content": user_content})

        completion = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.1,
            max_tokens=2048,
        )
        answer = completion.choices[0].message.content

        if mode == "chat" and session_id:
            history = chat_histories[session_id]
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            if len(history) > MAX_HISTORY * 2:
                chat_histories[session_id] = history[-MAX_HISTORY * 2:]

        return jsonify({"answer": answer, "ok": True, "engine": engine_key})

    except Exception as e:
        err = traceback.format_exc()
        print("ERROR:", err, flush=True)
        return jsonify({"error": "Error", "detail": str(e)}), 500


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
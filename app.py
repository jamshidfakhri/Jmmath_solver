import os
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from groq import Groq
from openai import OpenAI

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
) if OPENROUTER_API_KEY else None

# ============ Engines ============
ENGINES = {
    "lightning": {
        "name": "Lightning",
        "provider": "groq",
        "text_model": "openai/gpt-oss-120b",
        "vision_model": "qwen/qwen3.6-27b",
    },
    "reasoning": {
        "name": "Reasoning",
        "provider": "groq",
        "text_model": "deepseek-r1-distill-llama-70b",
        "vision_model": "qwen/qwen3.6-27b",
    },
    "gemini": {
        "name": "Gemini",
        "provider": "openrouter",
        "text_model": "google/gemini-2.0-flash-exp:free",
        "vision_model": "google/gemini-2.0-flash-exp:free",
    },
    "deepseek": {
        "name": "DeepSeek",
        "provider": "openrouter",
        "text_model": "deepseek/deepseek-chat:free",
        "vision_model": "deepseek/deepseek-chat:free",
    },
}

SYSTEM_PROMPT = """You are a mathematics solver. Your ONLY job is to solve math problems.

CRITICAL RULES:
1. Output ONLY the mathematical steps, one per line.
2. Do NOT write any explanation, no words, no sentences.
3. NO English, NO Persian, NO text at all.
4. Just show the equations and calculations step by step.
5. Use LaTeX math notation (e.g., \\frac{}{}, ^{}, \\sqrt{}, \\int, etc.).
6. Wrap each step in display math delimiters: $$ ... $$
7. At the end, write the final answer in a boxed format.

Example for input "2x + 5 = 15":
$$2x + 5 = 15$$
$$2x = 15 - 5$$
$$2x = 10$$
$$x = \\frac{10}{2}$$
$$x = 5$$

REMEMBER: No words. No explanations. Only math."""


def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/engines")
def engines():
    result = []
    for key, val in ENGINES.items():
        available = False
        if val["provider"] == "groq" and groq_client:
            available = True
        elif val["provider"] == "openrouter" and openrouter_client:
            available = True
        result.append({
            "key": key,
            "name": val["name"],
            "provider": val["provider"],
            "available": available
        })
    return jsonify({"engines": result})


@app.route("/api/solve", methods=["POST"])
def solve():
    try:
        question = request.form.get("question", "").strip()
        engine_key = request.form.get("engine", "lightning").strip()
        image = request.files.get("image")

        if not question and not image:
            return jsonify({"error": "Provide a question or an image"}), 400

        engine = ENGINES.get(engine_key)
        if not engine:
            return jsonify({"error": "Unknown engine"}), 400

        # ============ انتخاب کلاینت ============
        if engine["provider"] == "groq":
            if not groq_client:
                return jsonify({"error": "Groq API key not configured"}), 500
            client = groq_client
        elif engine["provider"] == "openrouter":
            if not openrouter_client:
                return jsonify({"error": "OpenRouter API key not configured"}), 500
            client = openrouter_client
        else:
            return jsonify({"error": "Unknown provider"}), 500

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if image and image.filename:
            img_b64 = encode_image(image)
            mime = image.mimetype or "image/jpeg"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question if question else "Solve this math problem."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                ]
            })
            model = engine["vision_model"]
        else:
            messages.append({"role": "user", "content": question})
            model = engine["text_model"]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )

        answer = completion.choices[0].message.content
        return jsonify({"answer": answer, "ok": True, "engine": engine_key})

    except Exception as e:
        error_detail = traceback.format_exc()
        print("ERROR DETAIL:", error_detail, flush=True)
        return jsonify({
            "error": "Error while solving",
            "detail": str(e),
            "trace": error_detail[-500:]
        }), 500


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
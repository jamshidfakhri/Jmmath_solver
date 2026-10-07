import os
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"

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

Example for input "انتگرال x^2":
$$\\int x^{2} \\, dx$$
$$= \\frac{x^{3}}{3} + C$$

REMEMBER: No words. No explanations. Only math."""


def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/solve", methods=["POST"])
def solve():
    if not client:
        return jsonify({"error": "API key not configured"}), 500

    try:
        question = request.form.get("question", "").strip()
        image = request.files.get("image")

        if not question and not image:
            return jsonify({"error": "سوال یا عکس رو وارد کن"}), 400

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
            model = VISION_MODEL
        else:
            messages.append({"role": "user", "content": question})
            model = TEXT_MODEL

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )

        answer = completion.choices[0].message.content
        return jsonify({"answer": answer, "ok": True})

    except Exception as e:
        error_detail = traceback.format_exc()
        print("ERROR DETAIL:", error_detail, flush=True)
        return jsonify({
            "error": "خطا در پردازش سوال",
            "detail": str(e),
            "trace": error_detail[-500:]
        }), 500


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
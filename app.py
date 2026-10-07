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

SYSTEM_PROMPT = """You are a professional mathematics tutor. Your job is to solve math problems step by step.

Rules:
1. Detect the language of the user's question (Persian or English) and answer in the SAME language.
2. If the question is in Persian, answer in Persian. If in English, answer in English.
3. Show the solution step by step, clearly.
4. Use LaTeX-like formatting for equations when helpful.
5. If the user sends an image, read the math problem from the image, then solve it.
6. If the problem is unclear, ask for clarification.
7. Be friendly and encouraging.

For Persian users:
- Write explanations in Persian.
- Math formulas can stay in Latin.

Always provide a clear final answer at the end."""


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
                    {"type": "text", "text": question if question else "این مسئله ریاضی رو حل کن."},
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
            temperature=0.3,
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
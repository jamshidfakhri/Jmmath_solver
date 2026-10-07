import os
import re
import ast
import math
import time
import threading
import logging
import operator as op
import urllib.request
import urllib.parse
from collections import OrderedDict
from flask import Flask, render_template, request, jsonify, Response
from groq import Groq
from openai import OpenAI

# ==================== Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("solver")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "solver-secret-xyz")

SITE_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://jmmath-solver.onrender.com")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
ADMIN_ID = os.environ.get("ADMIN_ID", "").strip()

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set.")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set.")
if not BOT_TOKEN or not ADMIN_ID:
    logger.info("Telegram feedback notifications disabled.")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
) if OPENROUTER_API_KEY else None

ENGINES = {
    "lightning": {"provider": "groq", "model": "llama-3.3-70b-versatile", "label": "LIGHTNING"},
    "reasoning": {"provider": "groq", "model": "deepseek-r1-distill-llama-70b", "label": "REASONING"},
    "gemini": {"provider": "openrouter", "model": "google/gemini-2.0-flash-exp:free", "label": "GEMINI"},
    "deepseek": {"provider": "openrouter", "model": "deepseek/deepseek-chat:free", "label": "DEEPSEEK"},
}

FALLBACK_ORDER = ["lightning", "gemini", "deepseek"]

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

# ==================== Feedback Stats ====================
_feedback_lock = threading.Lock()
feedback_stats = {
    "total": 0,
    "good": 0,
    "bad": 0,
    "recent": [],
}


# ==================== Safe Calculator ====================
SAFE_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
}
SAFE_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}
SAFE_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "min": min, "max": max, "pow": pow,
}
SAFE_CONSTS = {"pi": math.pi, "e": math.e}

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def normalize_text(text):
    if not text:
        return ""
    out = []
    for ch in text:
        if ch in PERSIAN_DIGITS:
            out.append(str(PERSIAN_DIGITS.index(ch)))
        elif ch in ARABIC_DIGITS:
            out.append(str(ARABIC_DIGITS.index(ch)))
        elif ch == "×":
            out.append("*")
        elif ch == "÷":
            out.append("/")
        elif ch in ("−", "–", "—"):
            out.append("-")
        elif ch == "٪":
            out.append("%")
        else:
            out.append(ch)
    s = "".join(out).replace("^", "**")
    return re.sub(r"\s+", " ", s).strip().lower()


def _check_node(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.BinOp):
        return type(node.op) in SAFE_BINOPS and _check_node(node.left) and _check_node(node.right)
    if isinstance(node, ast.UnaryOp):
        return type(node.op) in SAFE_UNARYOPS and _check_node(node.operand)
    if isinstance(node, ast.Call):
        return (isinstance(node.func, ast.Name)
                and node.func.id in SAFE_FUNCS
                and all(_check_node(a) for a in node.args))
    if isinstance(node, ast.Name):
        return node.id in SAFE_CONSTS
    return False


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINOPS:
        return SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARYOPS:
        return SAFE_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCS:
        return SAFE_FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in SAFE_CONSTS:
        return SAFE_CONSTS[node.id]
    raise ValueError("Not allowed")


def try_calculate(normalized_text):
    text = normalized_text.rstrip(" =?").strip()
    if not text:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)$", text)
    if m:
        try:
            return (text, float(m.group(1)) / 100.0 * float(m.group(2)))
        except Exception:
            return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    try:
        if not _check_node(tree.body):
            return None
        result = _safe_eval(tree.body)
        if isinstance(result, complex):
            return None
        return (text, result)
    except Exception:
        return None


def format_calc_answer(expr, result):
    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 10)
    display = expr.replace("**", "@@POW@@").replace("*", " \\times ").replace("@@POW@@", "^")
    return "$$" + display + " = " + str(result) + "$$"


# ==================== Cache ====================
_cache_lock = threading.Lock()
_cache = OrderedDict()
CACHE_MAX = 500
CACHE_TTL = 24 * 60 * 60


def cache_key(question, engine):
    return normalize_text(question) + "||" + engine


def cache_get(key):
    with _cache_lock:
        item = _cache.get(key)
        if item is None:
            return None
        ts, answer = item
        if time.time() - ts > CACHE_TTL:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return answer


def cache_set(key, answer):
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
        _cache[key] = (time.time(), answer)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


# ==================== Rate Limiter ====================
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 20
_rate_lock = threading.Lock()
_rate_store = {}


def get_client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_rate_limit(ip):
    now = time.time()
    with _rate_lock:
        times = [t for t in _rate_store.get(ip, []) if now - t < RATE_LIMIT_WINDOW]
        if len(times) >= RATE_LIMIT_MAX:
            retry = int(RATE_LIMIT_WINDOW - (now - times[0])) + 1
            _rate_store[ip] = times
            return (False, retry)
        times.append(now)
        _rate_store[ip] = times
        if len(_rate_store) > 5000:
            for k in list(_rate_store.keys()):
                _rate_store[k] = [t for t in _rate_store[k] if now - t < RATE_LIMIT_WINDOW]
                if not _rate_store[k]:
                    del _rate_store[k]
        return (True, 0)


# ==================== Fallback Chain ====================
def build_fallback_chain(selected):
    chain = [selected]
    for e in FALLBACK_ORDER:
        if e not in chain:
            chain.append(e)
    return chain


def call_ai_engine(engine_key, question, mode, session_id):
    engine = ENGINES[engine_key]

    if engine["provider"] == "groq":
        if not groq_client:
            raise RuntimeError("Groq client not initialized")
        client = groq_client
    elif engine["provider"] == "openrouter":
        if not openrouter_client:
            raise RuntimeError("OpenRouter client not initialized")
        client = openrouter_client
    else:
        raise RuntimeError("Unknown provider")

    if mode == "chat" and session_id:
        history = chat_histories.setdefault(session_id, [])
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    else:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    msgs.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model=engine["model"],
        messages=msgs,
        temperature=0.1,
        max_tokens=800,
        timeout=20.0,
    )
    return completion.choices[0].message.content


def send_telegram_notification(text):
    if not BOT_TOKEN or not ADMIN_ID:
        return
    try:
        url = "https://api.telegram.org/bot" + BOT_TOKEN + "/sendMessage"
        data = urllib.parse.urlencode({"chat_id": ADMIN_ID, "text": text[:4000]}).encode()
        urllib.request.urlopen(url, data=data, timeout=5)
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")


# ==================== Routes ====================
@app.route("/")
def home():
    return render_template("index.html", site_url=SITE_URL)


@app.route("/about")
def about():
    return render_template("about.html", site_url=SITE_URL)


@app.route("/guide")
def guide():
    return render_template("guide.html", site_url=SITE_URL)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


@app.route("/stats")
def stats():
    with _feedback_lock:
        return jsonify({
            "feedback": {
                "total": feedback_stats["total"],
                "good": feedback_stats["good"],
                "bad": feedback_stats["bad"],
                "ratio": round(
                    feedback_stats["good"] / feedback_stats["total"], 3
                ) if feedback_stats["total"] > 0 else None,
                "recent": feedback_stats["recent"][:10],
            }
        })


@app.route("/robots.txt")
def robots():
    txt = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /api/\n"
        "Disallow: /stats\n"
        "\n"
        f"Sitemap: {SITE_URL}/sitemap.xml\n"
    )
    return Response(txt, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    today = time.strftime("%Y-%m-%d")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f'  <url><loc>{SITE_URL}/</loc><lastmod>{today}</lastmod><changefreq>weekly</changefreq><priority>1.0</priority></url>\n'
        f'  <url><loc>{SITE_URL}/about</loc><lastmod>{today}</lastmod><changefreq>monthly</changefreq><priority>0.6</priority></url>\n'
        f'  <url><loc>{SITE_URL}/guide</loc><lastmod>{today}</lastmod><changefreq>monthly</changefreq><priority>0.7</priority></url>\n'
        '</urlset>\n'
    )
    return Response(xml, mimetype="application/xml")


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
        out.append({"key": key, "provider": val["provider"], "available": available, "label": val["label"]})
    return jsonify({"engines": out})


@app.route("/api/clear_chat", methods=["POST"])
def clear_chat():
    data = request.get_json() or {}
    sid = data.get("session_id", "")
    if sid in chat_histories:
        del chat_histories[sid]
    return jsonify({"ok": True})


@app.route("/api/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json() or {}
        rating = (data.get("rating") or "").strip().lower()
        question = (data.get("question") or "").strip()[:300]
        answer = (data.get("answer") or "").strip()[:600]
        engine_used = (data.get("engine_used") or "").strip()[:40]
        source = (data.get("source") or "").strip()[:40]

        if rating not in ("good", "bad"):
            return jsonify({"error": "Invalid rating"}), 400

        with _feedback_lock:
            feedback_stats["total"] += 1
            if rating == "good":
                feedback_stats["good"] += 1
            else:
                feedback_stats["bad"] += 1
            feedback_stats["recent"].insert(0, {
                "rating": rating,
                "question": question,
                "answer": answer[:200],
                "engine": engine_used or source,
                "source": source,
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            feedback_stats["recent"] = feedback_stats["recent"][:20]

        if rating == "bad":
            msg = (
                f"👎 Negative feedback on THE SOLVER\n\n"
                f"Question: {question}\n"
                f"Engine: {engine_used or source}\n\n"
                f"Answer (excerpt):\n{answer[:300]}"
            )
            send_telegram_notification(msg)

        return jsonify({"ok": True})

    except Exception:
        logger.exception("Feedback error")
        return jsonify({"error": "Error"}), 500


@app.route("/api/solve", methods=["POST"])
def solve():
    try:
        raw_question = request.form.get("question", "").strip()
        engine_key = request.form.get("engine", "lightning").strip()
        mode = request.form.get("mode", "solve").strip()
        session_id = request.form.get("session_id", "").strip()

        if not raw_question:
            return jsonify({"error": "Type your problem first."}), 400
        if len(raw_question) > 500:
            return jsonify({"error": "Question too long. Max 500 characters."}), 400

        engine = ENGINES.get(engine_key)
        if not engine:
            return jsonify({"error": "Unknown engine"}), 400

        normalized = normalize_text(raw_question)
        source = "ai"
        answer = None
        ckey = None
        engine_used = engine_key

        if mode == "solve":
            calc = try_calculate(normalized)
            if calc is not None:
                expr, result = calc
                answer = format_calc_answer(expr, result)
                source = "calculator"
                engine_used = None
            else:
                ckey = cache_key(raw_question, engine_key)
                cached = cache_get(ckey)
                if cached is not None:
                    answer = cached
                    source = "cache"
                    engine_used = None

        if answer is None:
            client_ip = get_client_ip()
            allowed, retry_after = check_rate_limit(client_ip)
            if not allowed:
                logger.warning(f"Rate limit hit for IP {client_ip}")
                return jsonify({
                    "error": "Too many requests. Please wait a moment.",
                    "rate_limited": True,
                    "retry_after": retry_after
                }), 429

            chain = build_fallback_chain(engine_key)
            last_error_type = None
            answer = None
            engine_used = None

            for ek in chain:
                start_ts = time.time()
                try:
                    result = call_ai_engine(ek, raw_question, mode, session_id)
                    if result and result.strip():
                        answer = result
                        engine_used = ek
                        source = "ai"
                        elapsed = round(time.time() - start_ts, 2)
                        logger.info(f"AI success: engine={ek} elapsed={elapsed}s mode={mode}")
                        break
                except Exception as e:
                    last_error_type = type(e).__name__
                    elapsed = round(time.time() - start_ts, 2)
                    logger.error(
                        f"AI failure: engine={ek} elapsed={elapsed}s "
                        f"error_type={last_error_type} error={str(e)[:200]}"
                    )
                    continue

            if answer is None:
                logger.error(f"All engines failed. last_error_type={last_error_type}")
                return jsonify({"error": "All engines are unavailable. Please try again in a moment."}), 503

            if mode == "chat" and session_id:
                history = chat_histories.setdefault(session_id, [])
                history.append({"role": "user", "content": raw_question})
                history.append({"role": "assistant", "content": answer})
                if len(history) > MAX_HISTORY * 2:
                    chat_histories[session_id] = history[-MAX_HISTORY * 2:]

            if mode == "solve" and ckey:
                cache_set(ckey, answer)

        return jsonify({
            "answer": answer,
            "ok": True,
            "engine": engine_key,
            "engine_used": engine_used,
            "engine_label": ENGINES[engine_used]["label"] if engine_used else None,
            "source": source
        })

    except Exception as e:
        logger.exception(f"Unhandled exception in /api/solve: {type(e).__name__}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
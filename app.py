import os
import re
import ast
import math
import time
import threading
import operator as op
import traceback
from collections import OrderedDict
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
    "lightning": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "reasoning": {"provider": "groq", "model": "deepseek-r1-distill-llama-70b"},
    "gemini": {"provider": "openrouter", "model": "google/gemini-2.0-flash-exp:free"},
    "deepseek": {"provider": "openrouter", "model": "deepseek/deepseek-chat:free"},
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

# ==================== Safe Calculator ====================
SAFE_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
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
    s = "".join(out)
    s = s.replace("^", "**")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _check_node(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.BinOp):
        if type(node.op) not in SAFE_BINOPS:
            return False
        return _check_node(node.left) and _check_node(node.right)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in SAFE_UNARYOPS:
            return False
        return _check_node(node.operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return False
        if node.func.id not in SAFE_FUNCS:
            return False
        return all(_check_node(a) for a in node.args)
    if isinstance(node, ast.Name):
        return node.id in SAFE_CONSTS
    return False


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers")
    if isinstance(node, ast.BinOp):
        t = type(node.op)
        if t not in SAFE_BINOPS:
            raise ValueError("BinOp not allowed")
        return SAFE_BINOPS[t](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        t = type(node.op)
        if t not in SAFE_UNARYOPS:
            raise ValueError("UnaryOp not allowed")
        return SAFE_UNARYOPS[t](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        fname = node.func.id
        if fname not in SAFE_FUNCS:
            raise ValueError("Func not allowed")
        args = [_safe_eval(a) for a in node.args]
        return SAFE_FUNCS[fname](*args)
    if isinstance(node, ast.Name):
        if node.id in SAFE_CONSTS:
            return SAFE_CONSTS[node.id]
        raise ValueError("Name not allowed")
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
        if result.is_integer():
            result = int(result)
        else:
            result = round(result, 10)
    display = expr.replace("**", "@@POW@@")
    display = display.replace("*", " \\times ")
    display = display.replace("@@POW@@", "^")
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
# فقط مسیر AI رو محدود می‌کنه، نه ماشین‌حساب و کش
RATE_LIMIT_WINDOW = 5 * 60      # ۵ دقیقه
RATE_LIMIT_MAX = 15              # ۱۵ درخواست AI در هر پنجره
_rate_lock = threading.Lock()
_rate_store = {}                 # {ip: [timestamps]}


def get_client_ip():
    """IP واقعی کاربر رو از هدرهای پروکسی می‌خونه."""
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_rate_limit(ip):
    """
    Returns:
      (allowed: bool, retry_after: int, remaining: int)
    """
    now = time.time()
    with _rate_lock:
        times = _rate_store.get(ip, [])
        # پاک‌سازی قدیمی‌ها
        times = [t for t in times if now - t < RATE_LIMIT_WINDOW]

        if len(times) >= RATE_LIMIT_MAX:
            oldest = times[0]
            retry = int(RATE_LIMIT_WINDOW - (now - oldest)) + 1
            _rate_store[ip] = times
            return (False, retry, 0)

        times.append(now)
        _rate_store[ip] = times
        remaining = RATE_LIMIT_MAX - len(times)

        # هر چند وقت یه بار، IPهای قدیمی رو پاک کن
        if len(_rate_store) > 5000:
            for k in list(_rate_store.keys()):
                _rate_store[k] = [t for t in _rate_store[k] if now - t < RATE_LIMIT_WINDOW]
                if not _rate_store[k]:
                    del _rate_store[k]

        return (True, 0, remaining)


# ==================== Routes ====================
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
        raw_question = request.form.get("question", "").strip()
        engine_key = request.form.get("engine", "lightning").strip()
        mode = request.form.get("mode", "solve").strip()
        session_id = request.form.get("session_id", "").strip()

        if not raw_question:
            return jsonify({"error": "Type your problem first."}), 400

        if len(raw_question) > 1000:
            return jsonify({"error": "Question too long. Max 1000 characters."}), 400

        engine = ENGINES.get(engine_key)
        if not engine:
            return jsonify({"error": "Unknown engine"}), 400

        normalized = normalize_text(raw_question)
        source = "ai"
        answer = None
        ckey = None

        # ---------- Solve Mode: Calculator + Cache ----------
        if mode == "solve":
            calc = try_calculate(normalized)
            if calc is not None:
                expr, result = calc
                answer = format_calc_answer(expr, result)
                source = "calculator"
            else:
                ckey = cache_key(raw_question, engine_key)
                cached = cache_get(ckey)
                if cached is not None:
                    answer = cached
                    source = "cache"

        # ---------- Fallback to AI ----------
        if answer is None:
            # ⬇⬇⬇ Rate limit فقط برای مسیر AI
            client_ip = get_client_ip()
            allowed, retry_after, remaining = check_rate_limit(client_ip)

            if not allowed:
                return jsonify({
                    "error": "Too many requests. Please wait a moment.",
                    "rate_limited": True,
                    "retry_after": retry_after
                }), 429

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

            msgs.append({"role": "user", "content": raw_question})

            completion = client.chat.completions.create(
                model=engine["model"],
                messages=msgs,
                temperature=0.1,
                max_tokens=800,
                timeout=15.0,
            )
            answer = completion.choices[0].message.content
            source = "ai"

            if mode == "chat" and session_id:
                history = chat_histories[session_id]
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
            "source": source
        })

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
"""AI assistant grounded in the farmer's context. Needs ANTHROPIC_API_KEY (paid API, billed to the
site owner). Privacy: sends LGA/ward, crop stage and weather advisories -- never name, phone or GPS.
Stateless: the client resends recent history each turn; nothing is stored server-side."""
import json, socket, urllib.error, urllib.request
from typing import Callable, Optional

MODEL_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MAX_TOKENS = 700
MAX_HISTORY_MESSAGES = 16     # ~8 exchanges; keeps request small and the farmer's own bill predictable

SYSTEM = """You are the KanoFarm AI assistant for smallholder farmers across Nigeria.
Rules:
- Use simple, practical language and short answers (a few sentences, or a short numbered list for steps).
- Farming conditions, seasons and common crops differ a lot between northern and southern Nigeria (and between zones such as North West, North East, North Central, South West, South East and South South). Use the farmer's FARM CONTEXT (state, LGA, ward) to give area-appropriate answers, and say plainly when you are unsure how something applies to their specific area.
- If the farmer writes in Hausa, Yoruba, Igbo or Nigerian Pidgin, answer in that language only if you are confident; otherwise answer simply in English and say that full support for that language is still limited.
- Use the FARM CONTEXT provided (weather advisories, crop, days after planting). If information you would need is missing from the context, say so plainly instead of guessing.
- Never invent weather figures, statistics, prevalence rates, or specific local agronomic numbers you were not given.
- Never recommend a specific pesticide product, brand, dose, or registration number. Suggest integrated pest management first: prevention, cultural, mechanical, then biological control. Say chemical control needs a registered product, verified with an extension officer, following the product label.
- You cannot diagnose a plant from a text description alone. Give the 2-3 most likely possible causes, how to start telling them apart, and recommend using the app's Scan Plant feature or an extension officer for a confirmed diagnosis.
- Be honest about uncertainty. You do not replace qualified agricultural extension professionals.
- Keep replies focused on farming in Nigeria. If asked something unrelated and harmless, answer briefly and steer back."""

class AssistantError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind   # auth | rate_limit | overloaded | bad_request | network | timeout | empty | server


def build_context(profile: dict, farm: Optional[dict], crops: list, advisories: list) -> str:
    lines = [f"Language preference: {profile.get('language', 'en')}"]
    if farm:
        lines.append(f"Farm state: {farm.get('state') or 'unknown'}, LGA={farm.get('lga') or 'unknown'}, ward={farm.get('ward') or 'unknown'}")
    else:
        lines.append("No farm selected: weather and crop context unavailable.")
    for c in crops:
        lines.append(f"Crop: {c['name']}, planted {c['planting_date']}, {c['days_after_planting']} days after planting")
    for a in advisories:
        lines.append(f"Weather advisory: {a['message_key']} evidence={json.dumps(a['evidence'])}")
    return "FARM CONTEXT\n" + "\n".join(lines)


def build_request(model: str, context: str, history: list, message: str) -> dict:
    """history: [{'role': 'user'|'assistant', 'content': str}, ...] already validated and trimmed."""
    messages = [{"role": h["role"], "content": h["content"]} for h in history] + [{"role": "user", "content": message}]
    return {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM + "\n\n" + context, "messages": messages}


def _post(key: str, payload: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(MODEL_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:   # nosec - fixed https URL, no user-controlled host
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            etype = json.loads(body).get("error", {}).get("type", "")
        except (json.JSONDecodeError, AttributeError):
            etype = ""
        kind = {401: "auth", 403: "auth", 429: "rate_limit", 529: "overloaded"}.get(e.code)
        if kind is None:
            kind = "bad_request" if e.code == 400 else "server"
        raise AssistantError(kind, f"http {e.code} {etype}") from e
    except socket.timeout as e:
        raise AssistantError("timeout") from e
    except urllib.error.URLError as e:
        raise AssistantError("network", str(e.reason)) from e
    except (TimeoutError, OSError) as e:
        raise AssistantError("network", type(e).__name__) from e


def ask(api_key: str, payload: dict, post: Callable = _post) -> str:
    if not api_key:
        raise AssistantError("auth", "no key configured")
    resp = post(api_key, payload)
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text").strip()
    if not text:
        raise AssistantError("empty")
    return text


FRIENDLY = {
    "auth": "The assistant is not set up correctly on this server.",
    "rate_limit": "Too many people are using the assistant right now. Please try again in a minute.",
    "overloaded": "The assistant is busy right now. Please try again shortly.",
    "timeout": "The assistant took too long to reply. Please try again.",
    "network": "No connection to the assistant right now. Please check your connection and try again.",
    "bad_request": "We couldn't send that question to the assistant. Please rephrase and try again.",
    "empty": "The assistant did not return an answer. Please try again.",
    "server": "The assistant had a temporary problem. Please try again shortly.",
}
# HTTP status this app returns to the farmer for each AssistantError kind
STATUS = {"auth": 503, "rate_limit": 429, "overloaded": 503, "timeout": 504,
          "network": 503, "bad_request": 422, "empty": 502, "server": 502}

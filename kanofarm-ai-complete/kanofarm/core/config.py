import os

def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

SUPABASE_URL = lambda: env("SUPABASE_URL").rstrip("/")
SUPABASE_ANON_KEY = lambda: env("SUPABASE_ANON_KEY")
AI_MODEL_ENDPOINT = lambda: env("AI_MODEL_ENDPOINT")
AI_MODEL_TOKEN = lambda: env("AI_MODEL_TOKEN")
ANTHROPIC_API_KEY = lambda: env("ANTHROPIC_API_KEY")
ASSISTANT_MODEL = lambda: env("ASSISTANT_MODEL", "claude-haiku-4-5-20251001")

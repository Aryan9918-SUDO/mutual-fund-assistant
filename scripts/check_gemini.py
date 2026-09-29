"""Quick diagnostic: is Gemini wired up correctly?

Run:  python scripts/check_gemini.py

Resolves your key from GEMINI_API_KEY (environment) or .streamlit/secrets.toml, makes one
tiny live call, and prints a clear SUCCESS / FAILURE with the exact reason. Use this to
confirm your key before launching the app.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mf_assistant import config  # noqa: E402


def resolve_key() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        m = re.search(r'GEMINI_API_KEY\s*=\s*"([^"]+)"', secrets.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return None


def main() -> int:
    key = resolve_key()
    if not key or key == "your-gemini-api-key-here":
        print("❌ No API key found.")
        print("   Set it one of these ways, then re-run:")
        print('   • export GEMINI_API_KEY="your-key"        (terminal / API / CLI)')
        print("   • put it in .streamlit/secrets.toml        (for the Streamlit app)")
        return 1

    print(f"🔑 Key found (…{key[-4:]}). Model: {config.GEMINI_MODEL}")
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        resp = model.generate_content("Reply with exactly: OK")
        text = (resp.text or "").strip()
        print(f"✅ Gemini responded: {text!r}")
        print("   Integration works. Launch the app and answers will use Gemini phrasing.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"❌ Gemini call failed: {type(e).__name__}: {e}")
        print("   Common causes: wrong/expired key, billing/quota, or a bad model name.")
        print("   Try a different model, e.g.  export GEMINI_MODEL=gemini-1.5-flash")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

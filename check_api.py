"""
Minimal connectivity check -- run this before touching the real pipeline
to confirm your API key and billing are set up correctly.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-your-key-here
    python check_api.py
"""

import os
import sys

import anthropic

def main():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY is not set in this shell.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-your-key-here")
        sys.exit(1)

    print(f"Found API key (starts with {key[:12]}...). Sending a test call...")

    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheapest model, fine for a connectivity check
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with just the word: connected"}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        print(f"Response: {text}")
        print("\nAPI access confirmed. You're ready to run the real pipeline.")
    except anthropic.AuthenticationError:
        print("Authentication failed -- check that your API key is correct and active.")
        sys.exit(1)
    except anthropic.PermissionDeniedError:
        print("Permission denied -- check billing is set up in the Console.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

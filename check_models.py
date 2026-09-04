"""
check_models.py

Run this once to see exactly which Groq models YOUR API key can access
right now. Model availability changes over time and by account, so this
is more reliable than any list in a tutorial or doc page.

Usage:
    python -m pipenv run python check_models.py
"""

import os
import requests

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found. Check your .env file exists and has GROQ_API_KEY=... in it.")
else:
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    print(f"Status code: {response.status_code}\n")

    if response.status_code == 200:
        data = response.json()
        print("Models available to your key:\n")
        for model in data.get("data", []):
            print(" -", model["id"])
    else:
        print("Something's wrong with the request itself:")
        print(response.text)
# import os
# from dotenv import load_dotenv
# from google import genai   # pip install google-genai

# load_dotenv()
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# print("Available models:")
# for m in client.models.list():
#     print(" ", m.name)

# MODEL = "gemini-3.5-flash"   # change to a name from the list above
# resp = client.models.generate_content(model=MODEL, contents="Say hello in Arabic")
# print(resp.text)

# import os, requests
# from dotenv import load_dotenv
# load_dotenv()

# r = requests.get(
#     "https://integrate.api.nvidia.com/v1/models",
#     headers={"Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY_EVAL')}"},
# )
# for m in r.json()["data"]:
#     print(m["id"])

# import os, requests
# from dotenv import load_dotenv
# load_dotenv()

# key = os.getenv("NVIDIA_API_KEY_EVAL")
# print("key loaded:", bool(key), (key or "")[:8])

# r = requests.get(
#     "https://integrate.api.nvidia.com/v1/models",
#     headers={"Authorization": f"Bearer {key}"},
# )
# print(r.status_code)
# print([m["id"] for m in r.json().get("data", [])][:50] if r.ok else r.text)

# r = requests.post(
#     "https://integrate.api.nvidia.com/v1/chat/completions",
#     headers={"Authorization": f"Bearer {key}"},
#     json={"model": "openai/gpt-oss-20b",
#           "messages": [{"role": "user", "content": "hi"}],
#           "max_tokens": 20},
# )
# print(r.status_code, r.text[:300])

import os

import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("NVIDIA_API_KEY_EVAL")
print("key", key)
for model in [
    "mistralai/mistral-large-2-instruct",
    "google/gemma-4-31b-it",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
]:
    print("model", model)
    r = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "قل مرحبا"}],
            "max_tokens": 30,
        },
        timeout=60,
    )
    print(model, r.status_code, r.text[:120])

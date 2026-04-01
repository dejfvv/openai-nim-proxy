from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Set your NVIDIA API key as an environment variable
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
# Example NIM endpoint — replace with your model's actual endpoint
NIM_URL = "[integrate.api.nvidia.com](https://integrate.api.nvidia.com/v1/chat/completions)"

@app.route("/v1/chat/completions", methods=["POST"])
def proxy_to_nvidia():
    data = request.json

    # Extract only the needed fields from the OpenAI-style request
    model = data.get("model", "meta/llama3-70b-instruct")
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 512)

    # Reformat request for NVIDIA API
    nim_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    # Send the request to NVIDIA NIM
    nim_response = requests.post(NIM_URL, json=nim_payload, headers=headers)
    if nim_response.status_code != 200:
        return jsonify({
            "error": {
                "message": f"NIM API error: {nim_response.text}",
                "type": "api_error"
            }
        }), nim_response.status_code

    nim_data = nim_response.json()

    # Adapt NVIDIA format → OpenAI-style response
    response = {
        "id": nim_data.get("id", "nim-proxy"),
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": nim_data["choices"][0]["message"],
                "finish_reason": nim_data["choices"][0].get("finish_reason", "stop")
            }
        ]
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

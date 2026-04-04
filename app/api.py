"""Flask REST API for the Bangla Literature Q&A system.

Run from project root:
    python -m app.api

Endpoint:
    POST /ask
    Body: {"query": "your question here"}
    Response: {"question": "...", "answer": "..."}
"""

from flask import Flask, request, jsonify
from src.rag.generation import generate_answer

app = Flask(__name__)


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query") if data else None

    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        memory = []
        answer = generate_answer(query, memory)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({"question": query, "answer": answer})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

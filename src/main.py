import os
import jwt
from functools import wraps
from flask import Flask, request
from rag import RAG
from flask_cors import CORS
from dotenv import load_dotenv
from prompts import social_skills_prompt, technical_skills_prompt, interview_prompt
from agents import DEFAULT_AGENTS

load_dotenv()

app = Flask(__name__)

CORS(app)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
# TODO: implement agents


def verify_token(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization").split(" ")[1]
        if not token:
            return {"status": "error", "message": "No token provided"}
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY,
                                 algorithms=[JWT_ALGORITHM])
            user_id = payload["user_id"]
            return func(user_id, *args, **kwargs)
        except jwt.exceptions.DecodeError:
            return {"status": "error", "message": "Invalid token"}
    return wrapper


@app.route("/query", methods=["POST"])
@verify_token
def query(user_id):
    question = request.json["question"]
    response = RAG.query(question, user_id)
    return response


@app.route("/documents", methods=["POST"])
@verify_token
def add_documents(user_id):
    document_keys = request.json["document_keys"]
    RAG.add_documents("justask", user_id, document_keys)
    return {"status": "success"}


@app.route("/documents", methods=["DELETE"])
@verify_token
def clear_documents(user_id):
    documents_keys = request.json["documents_keys"]
    RAG.clear_documents(user_id, documents_keys)
    return {"status": "success"}


@app.route("/agents/<agent>", methods=["GET"])
@verify_token
def custom_query(user_id, agent):
    if agent == "social":
        custom_prompt = social_skills_prompt
    elif agent == "technical":
        custom_prompt = technical_skills_prompt
    elif agent == "interview":
        custom_prompt = interview_prompt

    response = RAG.generate_question(custom_prompt, user_id)
    return response


@app.route("/agents", methods=["GET"])
@verify_token
def agents(user_id):
    lang = request.args.get("lang")

    if not lang:
        lang = "en"

    if lang not in DEFAULT_AGENTS:
        return {"status": "error", "message": "Invalid language"}

    return DEFAULT_AGENTS[lang]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)

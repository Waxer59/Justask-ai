from flask import Flask, request
from rag import RAG
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# TODO: use jwt instead of user_id
# TODO: implement agents

@app.route("/query", methods=["POST"])
def query():
    question = request.json["question"]
    user_id = request.json["user_id"]
    response = RAG.query(question, user_id)
    return response

@app.route("/documents", methods=["POST"])
def add_documents():
    user_id = request.json["user_id"]
    RAG.add_documents("justask",user_id)
    return {"status": "success"}

@app.route("/clearDocuments", methods=["DELETE"])
def clear_documents():
    user_id = request.json["user_id"]
    documents_keys = request.json["documents_keys"]
    RAG.clear_documents(user_id, documents_keys)
    return {"status": "success"}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)

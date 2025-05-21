import os
import boto3
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from PyPDF2 import PdfReader

load_dotenv()

MISTRAL_EMBEDDING_DIMENSIONS = 1024

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        vector_type="dense",
        dimension=MISTRAL_EMBEDDING_DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development" if ENVIRONMENT == "dev" else "production"
        }
    )

index = pc.Index(PINECONE_INDEX_NAME)

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = PineconeVectorStore(embedding=embedding_model, index=index)

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

prompt = ChatPromptTemplate(
    [
        ("human", """
    You are a helpful and knowledgeable assistant. Use only the information provided in the context to answer the following question clearly, accurately, and helpfully.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Answer only using the information available in the context.
    - If the information is not in the context, honestly say you don't know or there is not enough information.
    - Respond in the same language the question is asked.
    - Give direct and precise answers without prefacing with phrases like "According to the context" or similar.
    - Be clear, concise, and provide enough detail to fully answer the question.
    - Maintain a professional and friendly tone.
    """),
    ]
)


class RAG:
    @staticmethod
    def load_documents(bucket_name, prefix):
        s3 = boto3.client("s3",
                          endpoint_url=R2_ENDPOINT,
                          aws_access_key_id=R2_ACCESS_KEY_ID,
                          aws_secret_access_key=R2_SECRET_ACCESS_KEY)
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        docs = []

        for obj in response.get("Contents", []):
            file_key = obj["Key"]
            if file_key.endswith("/"):
                continue

            user_id = file_key.split("/")[0]
            ext = os.path.splitext(file_key)[1].lower()

            file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            raw_bytes = file_obj["Body"].read()

            content = None
            if ext in [".txt", ".md"]:
                try:
                    content = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = raw_bytes.decode("latin1")

            elif ext == ".pdf":
                try:
                    from io import BytesIO
                    reader = PdfReader(BytesIO(raw_bytes))
                    pages_text = []
                    for page in reader.pages:
                        pages_text.append(page.extract_text())
                    content = "\n".join(pages_text)
                except Exception as e:
                    print(f"ERROR WHILE READING {file_key}: {e}")
                    continue

            else:
                # Ignore files with unsupported extensions
                continue

            if content:
                doc = Document(page_content=content,
                               metadata={"user_id": user_id, "file_key": file_key})
                docs.append(doc)

        return docs

    @staticmethod
    def clear_documents(user_id, documents_keys):
        vectorstore.delete(
            filter={"user_id": user_id, "file_key": {"$in": documents_keys}})

    @staticmethod
    def add_documents(bucket_name, user_id):
        raw_docs = RAG.load_documents(bucket_name, prefix=f"{user_id}/")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(raw_docs)
        vectorstore.add_documents(docs)

    @staticmethod
    def query(question: str, user_id: str):
        retrieved_docs = vectorstore.similarity_search(
            question, filter={"user_id": user_id})
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.invoke(
            {"question": question, "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

import os
import zipfile
import torch
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# === Setup ===
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Unzip FAISS Index ===
zip_path = "faiss_index_company_details.zip"
extract_folder = "faiss_index_company_details"

if not os.path.exists(extract_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

# === Embedding ===
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# === Load FAISS ===
db = FAISS.load_local(
    folder_path=extract_folder,
    embeddings=embedder,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 3})

# === LLM ===
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

# === Prompt ===
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant for company interview preparation.
Based on the following context and company details, answer the question clearly.

{context}

Question: {question}
Answer:
"""
)

# === QA Chain ===
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# === Streamlit UI ===
st.title("EnlistAI – Company Insights & Interview Prep")

company_name = st.text_input("Enter Company Name")
question = st.text_input("Optional: Custom instruction for the AI (leave blank to use defaults)")

if company_name:
    try:
        # 🔍 Fetch API Data
        details_url = f"https://enlistai.onrender.com/company/{company_name}"
        q_url = f"https://enlistai-interview-question.onrender.com/questions/{company_name}"

        company_data = requests.get(details_url).json()
        questions_data = requests.get(q_url).json()

        context = f"Company Details:\n{company_data}\n\nInterview Questions:\n{questions_data.get('Questions', [])}"

        # 🧠 RAG response
        user_query = question if question else "Give an overview of this company and its interview questions."
        answer = rag_chain.run({"context": context, "question": user_query})

        st.write(answer)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")

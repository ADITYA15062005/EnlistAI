import os
import requests
import torch
import zipfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 🔒 Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 📌 Streamlit UI setup
st.set_page_config(page_title="EnlistAI - Interview Prep", page_icon="🎯")
st.title("🎯 EnlistAI – Company Insights & Interview Prep")
st.markdown("---")

# 🧠 Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
st.success("🔎 Embedding model loaded")

# 📁 Unzip and Load FAISS index
zip_path = "faiss_index_interview_questions.zip"
faiss_dir = "faiss_index_interview_questions"

try:
    if not os.path.exists(faiss_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(faiss_dir)
        st.info("🗃️ FAISS index unzipped.")

    db = FAISS.load_local(
        folder_path=faiss_dir,
        embeddings=embedder,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    st.success("📁 FAISS index loaded")
except Exception as e:
    st.error(f"❌ Failed to load FAISS index: {e}")

# 🤖 Connect to Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
st.success("🤖 Groq LLM connected")

# 📜 Prompt template
template = """
You are a company insights assistant.

Here is what we know:

Company Details:
{company_info}

Interview Questions:
{interview_questions}

Now, answer this user question clearly and professionally:
{user_question}
"""

prompt = PromptTemplate(
    input_variables=["company_info", "interview_questions", "user_question"],
    template=template
)

# ⛓️ Chain: prompt | llm
chain = prompt | llm

# 🏢 UI Inputs
company_name = st.text_input("🏢 Enter Company Name")
user_question = st.text_input("❓ Ask your question about the company")

# 🚀 Main Logic
if company_name and user_question:
    try:
        # 1️⃣ Get company details
        info_url = f"https://enlistai.onrender.com/company/{company_name}"
        info = requests.get(info_url).json()
        company_info = "\n".join(f"{k}: {v}" for k, v in info.items())

        # 2️⃣ Get interview questions
        questions_url = f"https://enlistai-interview-question.onrender.com/questions/{company_name}"
        q_res = requests.get(questions_url).json()
        if isinstance(q_res, list):
            interview_questions = "\n".join(q_res)
        elif isinstance(q_res, dict):
            interview_questions = "\n".join(q_res.get("Questions", []))
        else:
            interview_questions = "No interview questions found."

        # 3️⃣ Call chain
        with st.spinner("🤖 Thinking..."):
            answer = chain.invoke({
                "company_info": company_info,
                "interview_questions": interview_questions,
                "user_question": user_question
            })

        # 4️⃣ Display
        st.markdown("### ✅ Answer")
        st.write(answer.content if hasattr(answer, "content") else answer)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")

import os

import streamlit as st

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# -------------------------
# Load Environment Variables
# -------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------------
# Streamlit Page
# -------------------------

st.set_page_config(
    page_title="Document Question Answering",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering using RAG")

st.write(
    "Ask questions about your uploaded documents."
)

# -------------------------
# Load Embedding Model
# -------------------------

@st.cache_resource
def load_embedding():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding()

# -------------------------
# Load Vector Store
# -------------------------

@st.cache_resource
def load_vector_db():

    db = FAISS.load_local(
        "vector_store",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db

vector_db = load_vector_db()

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":3,
        "fetch_k":10
    }
)

# -------------------------
# Load Groq
# -------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# -------------------------
# Prompt
# -------------------------

PROMPT = """
You are an intelligent Document Question Answering Assistant.

Rules:

1. Answer ONLY from the provided context.

2. Never use outside knowledge.

3. If the answer is unavailable, reply:

"I don't know. The uploaded documents do not contain this information."

Context:

{context}

Question:

{question}

Answer:
"""

# -------------------------
# Question Input
# -------------------------

question = st.text_input(
    "Enter your question"
)

# -------------------------
# Ask Button
# -------------------------

if st.button("Get Answer"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = PROMPT.format(
            context=context,
            question=question
        )

        response = llm.invoke(prompt)

        st.subheader("Answer")

        st.success(response.content)

        st.subheader("Retrieved Context")

        for i, doc in enumerate(docs, start=1):

            with st.expander(f"Chunk {i}"):

                st.write(doc.page_content)

                st.write("Metadata")

                st.json(doc.metadata)
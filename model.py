from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from util import chunk_text
import google.generativeai as genai
import os

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def build_faiss_index(text):
    chunks = chunk_text(text)

    if not chunks:
        return None, []

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, chunks


def ask_bot(question, index, chunks, k=3):
    q_embedding = model.encode([question])
    _, I = index.search(np.array(q_embedding), k)

    context = " ".join([chunks[i] for i in I[0]])
    return context


def generate_answer(context, question):
    prompt = f"""
Answer the question using ONLY the following content:

{context}

Question: {question}
"""
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content(prompt)
    return response.text


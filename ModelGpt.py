from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from util import chunk_text
from openai import OpenAI
import os

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load OpenAI client safely
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
print("API KEY_____________",client.api_key is not None)

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content



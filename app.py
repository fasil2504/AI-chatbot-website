import streamlit as st
from scraper import scrape_site
from model import build_faiss_index, ask_bot, generate_answer

st.title("🤖 AI Chatbot (Website)")

url = st.text_input("Website URL")
question = st.text_input("Ask a question")

if st.button("Ask"):
    text = scrape_site(url)

    index, chunks = build_faiss_index(text)
    context = ask_bot(question, index, chunks)

    final_answer = generate_answer(context, question)

    st.subheader("Answer")
    st.write(final_answer)

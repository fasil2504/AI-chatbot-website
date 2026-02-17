
from model import ask_bot, generate_answer
from model import build_faiss_index  
from scraper import scrape_site

url="https://www.icliniq.com/articles"
question = "what  type of company"
if question:
    text = scrape_site(url)
    index, chunks = build_faiss_index(text)
    context = ask_bot(question,index, chunks)
    answer = generate_answer(context, question)
    print(answer)
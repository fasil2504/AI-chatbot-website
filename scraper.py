import requests
from bs4 import BeautifulSoup

def scrape_site(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = " ".join(soup.get_text().split())
    return text

import requests
import streamlit as st
from langchain.schema import Document

def call_fmp_api(api_key):
    api_key
    url = f"https://financialmodelingprep.com/stable/fmp-articles?page=0&limit=1&apikey={api_key}"
    
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to fetch news articles.")
        return []

    articles = response.json()
    docs = []

    for article in articles:
        title = article.get("title", "")
        content = article.get("content", "")
        link = article.get("link", "")
        date = article.get("date", "")

        if content:
            metadata = {
                "source": link,
                "title": title,
                "date": date
            }
            docs.append(Document(page_content=content, metadata=metadata))

    return docs
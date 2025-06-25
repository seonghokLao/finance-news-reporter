import os
import streamlit as st
import pickle
import time
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage

from datetime import datetime, timedelta

from util import *

# Load environment variables
load_dotenv()

st.title("Finance News Reporter")
st.sidebar.title("Web Scraper")

# --- Sidebar Inputs ---
sites = ["All", "FMP", "newsdata.io"]
site = st.sidebar.selectbox("Sources", sites)

keyword = st.sidebar.text_input("Keywords (Optional)", value="")
summarize = st.sidebar.toggle("Summarize News")
submitted = st.sidebar.button("Submit")
faiss_path = "faiss_store/faiss_store.pkl"
main_placeholder = st.empty()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.3, max_tokens=500)

query = None

data = []

if submitted:

    if site in ["All", "FMP"]:
        main_placeholder.text("Scraping...")
        # since_date = (datetime.utcnow() - timedelta(days=0)).strftime('%Y-%m-%d')

        data += call_fmp_api(api_key=os.getenv("FMP_API_KEY"))

    # print(data)
    if not data:
        st.warning("No articles found.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Splitting Text....")
        chunks = text_splitter.split_documents(data)

        # print(chunks)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        main_placeholder.text("Building Embeddings...")

        with open(faiss_path, 'wb') as f:
            pickle.dump(vectorstore, f)

        st.success("News embedded successfully. You can start asking questions.")

# --- Summary Generation ---
if summarize and data:
    st.subheader("Summarized Articles:")
    for doc in data:
        print("Summarizing article")
        prompt = f"In two sentences, tell me the economic implications, if any, and which specific stock prices might be affected, if any.: {doc.page_content}"
        try:
            summary = llm([HumanMessage(content=prompt)]).content
            st.markdown(f"**{doc.metadata.get('title', 'Untitled')}**\n- {summary}")
        except Exception as e:
            st.warning(f"Failed to summarize article: {e}")

# --- Question Answering ---
query = main_placeholder.text_input("Question: ")

if query and os.path.exists(faiss_path):
    with open(faiss_path, "rb") as f:
        vectorstore = pickle.load(f)
        print("Retrieving Answer")
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        print("got answer")

        st.header("Answer:")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources = sources.split("\n")
            for source in sources:
                st.write(source)

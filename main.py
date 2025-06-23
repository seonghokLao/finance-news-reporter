import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("Finance News Reporter")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

submitted = st.sidebar.button("submit")
faiss_path = "faiss_store.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if submitted:
    # load articles
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading Articles...")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Text....")
    chunks = text_splitter.split_documents(data)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    main_placeholder.text("Building Embeddings...")
    
    with open(faiss_path, 'wb') as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")

if query and os.path.exists(faiss_path):
    with open(faiss_path, "rb") as f:
        vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.fromllm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)


        
from langchain_core.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import pandas as pd

def get_data(file_path):
    data = pd.read_json(file_path)
    return data

def load_documents(data):
    docs = []
    for idx, row in data.iterrows():
        doc = Document(page_content=row['article_text'], metadata={"url":row['url'], "title":row['title'], "author": row['author'], "publish_date": row['publish_date'], "article_text":row['article_text']})
        docs.append(doc)
    return docs

def retrieve_context(docs, chunking=True, chunk_size=800, chunk_overlap=150):
    if chunking:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=' ')
        docs = splitter.split_documents(docs)

    #store in a vector store
    vector_stores = FAISS.from_documents(documents= docs, embedding=OpenAIEmbeddings())

    #retreive relevant chunks using retriever
    retriever = vector_stores.as_retriever(search_kwargs={"k": 3})

    return retriever

    # final_context = retriever.invoke(query)

    # context = "\n".join([i.page_content for i in final_context])
    # return context
import os
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import torch
nest_asyncio.apply()
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Cohere
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

load_dotenv()

def main():
    st.header("Q&A BOT FOR PERSONAL PDF FILES")
    pdf = st.file_uploader("Upload your PDF here/-", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        embed_model = embed_model = CohereEmbeddings(model="embed-english-v3.0")
        vectorstore = FAISS.from_texts(chunks, embedding=embed_model)
        st.write('Embeddings Computed')

        query_ = st.text_input("Ask questions related to your PDF")
        st.write(query_)
        if query_:
            docs = vectorstore.similarity_search(query=query_, k=3)
            llm_ = Cohere(model="command-xlarge-nightly")
            chain = load_qa_chain(llm=llm_, chain_type="stuff")
            with get_openai_callback() as CB:
                response = chain.run(input_documents=docs, question=query_)
                print(CB)
                st.write(CB)
            st.write(response)
if __name__ == '__main__':
    main()
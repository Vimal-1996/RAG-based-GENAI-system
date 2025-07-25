# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 13:10:34 2025

@author: User
"""

import streamlit as st
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_KEY")
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['GROQ_KEY']=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=  os.getenv("MY_OPENAI_KEY")

##create an LLm model
llm = ChatGroq(model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only
    Please respond the most accurate response based on the question
    <context>
    {context}
    <context>
    
    Question:{input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        #Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader("papers")
        st.session_state.docs = st.session_state.loader.load()
        #Chunks
        st.session_state.textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.textsplitter.split_documents(st.session_state.docs[:50])
        #Embedding and Storing in vector database
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        #Setup the retriever
        
        #Integrate LLM
        
        #Build RAG chain
        
        #Deploy your application
        
if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vevtor database is ready")

user_prompt = st.text_input("Enter your query from research paper")

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please generate the vector database first by clicking the 'Document Embedding' button.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain) 
        response = retrieval_chain.invoke({"input":user_prompt})
        st.write(response["answer"] if "answer" in response else response)
    
    
    
    
    
    





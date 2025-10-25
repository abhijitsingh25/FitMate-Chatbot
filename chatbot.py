#from langchain.document_loaders import PyPDFLoader
#from langchain.indexes import VectorstoreIndexCreator
#from langchain.chains import retrieval_qa
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

#from ibm_watsonx_ai import langChainInterface

#title
st.title("Ask GimmY")
#display prompt
prompt = st.chat_input('Ask the question')

if prompt:
    st.chat_messege('user').markdown(prompt)

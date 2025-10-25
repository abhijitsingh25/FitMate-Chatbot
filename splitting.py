from langchain_text_splitters import RecursiveCharacterTextSplitter
from indexing import documents

def split_docs(documents, chunk_size=500, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

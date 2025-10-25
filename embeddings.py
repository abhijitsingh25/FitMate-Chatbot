from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from splitting import docs

import asyncio

# API Keys & Config
pinecone_api_key = "pcsk_3ryyUH_PSXqouQsYgpTRr78Ynme7CSJS4ocVL8ExscaJk4FMerw2RRmnhspQb2ryW6DYWf"
pinecone_environment = "us-east-1"  # Replace with your actual environment region
INDEX_NAME = "gymmy"

# 1. Initialize embedding model
embedding_model_used = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create or connect to Pinecone index
def initialize_pinecone(api_key, environment, index_name, embedding_model):
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_model._client.get_sentence_embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment)
        )
    return pc.Index(index_name)

# 3. Get vector store using LangChain
def get_pinecone_vectorstore(api_key, environment, index_name, embedding_model):
    index = initialize_pinecone(api_key, environment, index_name, embedding_model)
    vector_store = PineconeVectorStore(index, embedding_model, text_key="text")
    return vector_store

pinecone_vector_store = get_pinecone_vectorstore(
        pinecone_api_key, pinecone_environment, INDEX_NAME, embedding_model_used
    )

# 4. Optional async add_documents wrapper
def generate_embeddings(chunks):
    asyncio.run(pinecone_vector_store.aadd_documents(chunks))

generate_embeddings(docs)
# if __name__=="__main__":
#     asyncio.run(generate_embeddings(docs))


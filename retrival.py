from langchain_google_vertexai import ChatVertexAI
from langchain.chains import RetrievalQA
from embeddings import pinecone_vector_store
from vertexai import init as vertexai_init

vertexai_init(project="engaged-patrol-456216-u0", location="us-central1"),
# 1. Load Gemini model via Vertex AI
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",  # Latest Gemini 1.5 model
    temperature=0.3,
    max_output_tokens=1024,
    convert_system_message_to_human=True
)

# 2. Setup retrieval-based QA
qa_chain = RetrievalQA.from_chain_type(
    
    llm=llm,
    retriever=pinecone_vector_store.as_retriever(),
    return_source_documents=True
)

# 3. Ask a question
query = "What is the main idea of the book ?"
response = qa_chain.invoke(query)

print(response['result'])

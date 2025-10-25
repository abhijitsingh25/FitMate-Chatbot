import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from embeddings import pinecone_vector_store
from vertexai import init as vertexai_init
from tavily import TavilyClient
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Gymmy", layout="centered")

load_dotenv()

# --- Custom Prompt Template ---
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and expert fitness assistant.

You ONLY answer questions related to fitness — including exercise techniques, diet plans, pain or injuries, recovery, workout routines, and health optimization. If a question is not fitness-related, respond politely saying you are only designed to answer fitness-related queries.

Use the context below to answer the question, and DO NOT answer anything outside the scope of fitness. Be concise and clear.

Context:
{context}

Question:
{question}

Answer:
"""
)


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query):
    try:
        results = tavily_client.search(query=query, search_depth="basic", max_results=3)
        return "\n\n".join([r['content'] for r in results['results']])
    except Exception as e:
        return "No relevant web data found."



# --- Load Gemini + Retrieval Chain ---
@st.cache_resource
def load_qa_chain():
    vertexai_init(project="engaged-patrol-456216-u0", location="us-central1")

    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-001",
        temperature=0.3,
        max_output_tokens=1024,
        convert_system_message_to_human=True
    )

    qa_chain = RetrievalQA(
        retriever=pinecone_vector_store.as_retriever(),
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=custom_prompt),
            document_variable_name="context"
        ),
        return_source_documents=True
    )
    return qa_chain


qa_chain = load_qa_chain()

# --- Streamlit Setup ---
st.title("Gymmy: Your health Assistant")

# --- Session State for Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input Box ---
query = st.chat_input("Ask a question about the book...")

if query:
    with st.spinner("Thinking..."):
        # 1. Get book context
        pinecone_response = qa_chain.invoke(query)
        pinecone_context = "\n\n".join(
            doc.page_content for doc in pinecone_response['source_documents']
        )

        # 2. Get web context
        web_context = web_search(query)

        # 3. Combine both
        combined_context = f"{pinecone_context}\n\n---\n\n{web_context}"

        # 4. Ask Gemini again using combined context
        vertexai_init(project="engaged-patrol-456216-u0", location="us-central1")
        llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0.3)

        prompt = custom_prompt.format(context=combined_context, question=query)
        answer = llm.invoke(prompt).content

        # Save chat
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))

# --- Display Chat History ---
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

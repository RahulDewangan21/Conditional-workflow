import os
from typing import TypedDict, Annotated

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PAGE CONFIG (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="BIT Durg College Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — modern chat look
# ============================================================

st.markdown(
    """
    <style>
        /* Overall app background */
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        /* Main title block */
        .main-header {
            text-align: center;
            padding: 1.2rem 0 0.4rem 0;
        }
        .main-header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.1rem;
        }
        .main-header p {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 0;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.4rem 0.2rem;
        }

        /* Badge for query type */
        .query-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 10px;
            border-radius: 999px;
            margin-bottom: 6px;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .badge-academic { background: #1e3a8a; color: #bfdbfe; }
        .badge-fee { background: #713f12; color: #fde68a; }
        .badge-general { background: #14532d; color: #bbf7d0; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid #1f2937;
        }
        .sidebar-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .sidebar-sub {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 1.2rem;
        }

        /* Footer note */
        .footer-note {
            text-align: center;
            color: #64748b;
            font-size: 0.75rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STEP 1 — Build embeddings + retrievers (cached so this
# only runs ONCE, not on every Streamlit rerun)
# ============================================================


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="Indexing college documents...")
def load_retrievers(_embedding):
    def build_retriever(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(document)
        vectorstore = Chroma.from_documents(chunks, _embedding)
        return vectorstore.as_retriever(search_kwargs={"k": 2})

    academic = build_retriever(pdf_path="academic_regulations.pdf")
    fee = build_retriever(pdf_path="fee_structures.pdf")
    return academic, fee


embedding = load_embedding_model()
academic_retriever, fee_retriever = load_retrievers(embedding)



# ============================================================
# STEP 3 — LLM
# ============================================================

llm = ChatMistralAI(model="mistral-small-2603", temperature=0.4)
# ============================================================
# STEP 4 — State
# ============================================================


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ============================================================
# STEP 5 — Nodes
# ============================================================

def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content  #[-1] refers to last message

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    print("CLASSIFIER OUTPUT:", category)

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    print("FINAL CATEGORY:", category)

    return {"query_type" : category}


def route_query(state: State):
    query_type = state["query_type"]

    if query_type == "academic":
        return "academic_rag"
    elif query_type == "fee":
        return "fee_rag"
    else:
        return "general"


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}



def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    query_type = state.get("query_type", "general")
    context = state.get("retrieved_context", "NO_RETRIEVAL_NEEDED")

    # Case 1: General question, no retrieval needed
    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )

    # Case 2: Fee search happened but found nothing useful
    elif context == "NO_FEE_INFO_FOUND_ONLINE":
        prompt = (
            f"You are a college assistant. A {programme} student asked about fees, "
            f"but a search of the college website didn't return clear information "
            f"for this question: {query}\n\n"
            f"Politely tell the student that exact fee details aren't available online "
            f"right now, and suggest they check with the accounts office or the official "
            f"college website directly."
        )

    # Case 3: Fee info was scraped from the web (not an official document)
    elif query_type == "fee":
        prompt = (
            f"You are a college assistant helping a {programme} student with a fee question. "
            f"Use the following information found on the college website to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Important: This information may come from third-party estimates rather than "
            f"an official fee circular. Clearly tell the student these are approximate "
            f"figures and they should confirm exact amounts with the accounts office "
            f"before making any payment decision."
        )

    # Case 4: Academic info from your PDF (this one you trust)
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# ============================================================
# STEP 6 — Build graph (cached — build once per session)
# ============================================================


@st.cache_resource(show_spinner=False)
def build_app():
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges(
        "classifier",
        route_query,
        {
            "academic_rag": "academic_rag",
            "fee_rag": "fee_rag",
            "general": "general",
        },
    )

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


app = build_app()

# ============================================================
# STEP 7 — Streamlit session state
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {role, content, query_type}

if "programme" not in st.session_state:
    st.session_state.programme = "BTech"

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">🎓 BIT, Durg</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Bhilai Institute of Technology — AI Assistant</div>',
        unsafe_allow_html=True,
    )

    st.session_state.programme = st.selectbox(
        "Your Programme",
        options=["BTech", "MTech", "MBA"],
        index=["BTech", "MTech", "MBA"].index(st.session_state.programme),
    )

    st.markdown("---")
    st.markdown("**What I can help with:**")
    st.markdown("📘 Academic rules & regulations")
    st.markdown("💰 Fee-related questions")
    st.markdown("💬 General queries")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        '<div class="footer-note">Answers may be inaccurate.<br>Verify important info with the college office.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 BIT Durg College Assistant</h1>
        <p>Ask me about academics, fees, or anything else about college life</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# RENDER CHAT HISTORY
# ============================================================

badge_map = {
    "academic": ('<span class="query-badge badge-academic">📘 Academic</span>'),
    "fee": ('<span class="query-badge badge-fee">💰 Fee</span>'),
    "general": ('<span class="query-badge badge-general">💬 General</span>'),
}

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("query_type"):
            st.markdown(badge_map.get(msg["query_type"], ""), unsafe_allow_html=True)
        st.markdown(msg["content"])

# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(f"Ask something as a {st.session_state.programme} student...")

if user_query:
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke(
                {
                    "programme": st.session_state.programme,
                    "messages": [("human", user_query)],
                }
            )
            answer = result["messages"][-1].content
            query_type = result.get("query_type", "general")

        st.markdown(badge_map.get(query_type, ""), unsafe_allow_html=True)
        st.markdown(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "query_type": query_type}
    )

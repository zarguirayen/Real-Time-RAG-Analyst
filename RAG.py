import time
import json
import os
import threading
from datetime import datetime
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.docstore.document import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama

# ===============================
# 1. CONFIGURATION
# ===============================
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
STATIC_DB_PATH = "data/chroma_static"
LIVE_DB_PATH = "data/chroma_live"
FEED_FILE = "data/live_feed.json"
os.makedirs("data", exist_ok=True)

# Thread lock to avoid simultaneous write conflicts
file_lock = threading.Lock()

@st.cache_resource
def load_vector_stores():
    print("Loading static database (Collection: static_history)...")
    static_vectorstore = Chroma(
        persist_directory=STATIC_DB_PATH, 
        embedding_function=embeddings,
        collection_name="static_history"
    )
    
    print("Loading live database (Collection: live_stream)...")
    live_vectorstore = Chroma(
        persist_directory=LIVE_DB_PATH,
        embedding_function=embeddings,
        collection_name="live_stream"
    )
    print("Databases loaded successfully.")
    return static_vectorstore, live_vectorstore

static_vectorstore, live_vectorstore = load_vector_stores()

# ===============================
# 2. BACKGROUND TASK: [FEEDER]
# ===============================
def update_live_feed(vectorstore, embeddings):
    fake_news = [
        "TechCorp announces an ambitious decarbonization plan for 2030.",
        "TechCorp shares drop by 3% following a bug in a flagship product.",
        "TechCorp launches a new division dedicated to generative AI.",
        "Rumor: a strategic partnership between TechCorp and an automotive giant."
    ]

    with file_lock:
        if not os.path.exists(FEED_FILE):
            json.dump([], open(FEED_FILE, "w"))

        existing = json.load(open(FEED_FILE))
        existing_titles = [a["content"] for a in existing]
        news_to_add = None

        for news in fake_news:
            if news not in existing_titles:
                news_to_add = news
                break

        if news_to_add:
            print(f"\n[FEEDER] New news detected: {news_to_add}")
            new_doc = Document(
                page_content=news_to_add, 
                metadata={"source": "Simulated Feed", "timestamp": datetime.now().isoformat()}
            )
            vectorstore.add_documents([new_doc])
            vectorstore.persist()

            existing.append({
                "timestamp": datetime.now().isoformat(),
                "content": news_to_add
            })
            json.dump(existing, open(FEED_FILE, "w"), indent=2, ensure_ascii=False)
            print("[FEEDER] Live database updated and persisted.")
        else:
            print("[FEEDER] All news have been processed. Resetting feed.")
            json.dump([], open(FEED_FILE, "w"))

def feed_updater_task():
    print(" Background task [FEEDER] started...")
    while True:
        update_live_feed(live_vectorstore, embeddings)
        time.sleep(15)

# ===============================
# 3. THE [ANALYST] (RAG CHAIN)
# ===============================
@st.cache_resource
def create_rag_chain(_static_store, _live_store):
    print("Setting up RAG chain [ANALYST] (Prompt v4)...")

    def get_all_static_context(question_input):
        docs = _static_store.get(include=["documents"])['documents']
        return "\n".join(docs) if docs else "No historical context available."

    def get_all_live_context(question_input):
        docs = _live_store.get(include=["documents"])['documents']
        return "\n".join(docs) if docs else "No 'live' information available."

    try:
        llm = ChatOllama(model="mistral")
        llm.invoke("Test")
        print("Successfully connected to Ollama.")
    except Exception as e:
        st.error(f"ERROR: Unable to connect to Ollama. {e}")
        return None

    template = """
You are an expert market analyst. Follow the instructions exactly.
Do NOT invent information. Base your answer ONLY on the provided contexts.

--- START Historical Context (Reliable) ---
{context_static}
--- END Historical Context ---

--- START Live Context (Recent) ---
{context_live}
--- END Live Context ---

User Question: {question}

Response Instructions (FOLLOW THESE 3 STEPS):

Step 1: Summarize the 'Live' facts
(List the key facts from the 'Live Context'.)

Step 2: Recall the historical context
(Cite the relevant historical facts exactly as they appear.)

Step 3: Analysis
(Answer the user's question, using facts from Steps 1 and 2 to justify your reasoning.)
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain_setup = RunnableParallel({
        "context_static": RunnableLambda(get_all_static_context),
        "context_live": RunnableLambda(get_all_live_context),
        "question": RunnablePassthrough()
    })

    final_chain = chain_setup | prompt | llm | StrOutputParser()
    print("RAG chain [ANALYST] ready.")
    return final_chain

# ===============================
# 4. STREAMLIT WEB INTERFACE
# ===============================
def main():
    st.set_page_config(page_title="Real-Time RAG Analyst", layout="wide")
    st.title("🤖 Real-Time RAG Analyst")

    st.sidebar.title("Live News Feed")
    st.sidebar.markdown("The [FEEDER] adds one news item every 15s.")
    sidebar_feed = st.sidebar.empty()

    def display_live_feed():
        with file_lock:
            if not os.path.exists(FEED_FILE):
                sidebar_feed.write("Waiting for first news...")
                return
            
            try:
                feed_data = json.load(open(FEED_FILE))
            except json.JSONDecodeError:
                return  # wait for next cycle
                
            sidebar_feed.empty()
            if not feed_data:
                sidebar_feed.write("Feed is resetting...")
                return

            for item in reversed(feed_data[-5:]):
                with st.sidebar.expander(f"_{item['content'][:40]}..._"):
                    st.markdown(f"**Content:** {item['content']}")
                    st.caption(f"Timestamp: {item['timestamp']}")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask me a question about TechCorp."}
        ]

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = create_rag_chain(static_vectorstore, live_vectorstore)

    if "feeder_started" not in st.session_state:
        print("Starting [FEEDER] in background...")
        feeder_thread = threading.Thread(target=feed_updater_task, daemon=True)
        feeder_thread.start()
        st.session_state.feeder_started = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    display_live_feed()

    if query := st.chat_input("What is the current situation of TechCorp?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        greetings = ["hi", "hello", "bonjour", "salut", "yo", "hey"]
        is_greeting = any(greet in query.lower() for greet in greetings)
        
        if is_greeting:
            response = (
                "Hello! I’m a RAG analyst. I can only answer factual questions about TechCorp "
                "based on the data provided."
            )
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("...")

                if st.session_state.rag_chain:
                    try:
                        response = st.session_state.rag_chain.invoke(query)
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error while calling Ollama: {e}")
                else:
                    st.error("Error: RAG chain not initialized.")

if __name__ == "__main__":
    main()

# streamlit run RAG.py    
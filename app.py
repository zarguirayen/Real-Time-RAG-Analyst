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

# --- 1. Configuration ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
STATIC_DB_PATH = "data/chroma_static"
LIVE_DB_PATH = "data/chroma_live"
FEED_FILE = "data/live_feed.json"
os.makedirs("data", exist_ok=True)

# --- MODIFICATION (v13): Add Thread Lock ---
file_lock = threading.Lock()
# --- END MODIFICATION ---

@st.cache_resource
def load_vector_stores():
    print("Loading static database (Collection: static_history)...")
    static_vectorstore = Chroma(
        persist_directory=STATIC_DB_PATH,
        embedding_function=embeddings,
        collection_name="static_history"
    )

    print("Loading 'live' database (Collection: live_stream)...")
    live_vectorstore = Chroma(
        persist_directory=LIVE_DB_PATH,
        embedding_function=embeddings,
        collection_name="live_stream"
    )
    print("Databases loaded.")
    return static_vectorstore, live_vectorstore

static_vectorstore, live_vectorstore = load_vector_stores()

# --- 2. Background Task: The [FEEDER] ---
def update_live_feed(vectorstore, embeddings):

    fake_news = [
        "TechCorp announces an ambitious decarbonization plan for 2030.",
        "TechCorp shares fall by 3% after a bug in a flagship product.",
        "TechCorp launches a division dedicated to generative artificial intelligence.",
        "Rumor: a strategic partnership between TechCorp and a major automotive company."
    ]

    # --- MODIFICATION (v13): Use Lock ---
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
            print(f"\n[FEEDER]  New item detected: {news_to_add}")
            new_doc = Document(
                page_content=news_to_add,
                metadata={"source": "Simulated Feed", "timestamp": datetime.now().isoformat()}
            )
            vectorstore.add_documents([new_doc])
            vectorstore.persist()
            existing.append({"timestamp": datetime.now().isoformat(), "content": news_to_add})
            json.dump(existing, open(FEED_FILE, "w"), indent=2, ensure_ascii=False)
            print("[FEEDER] 'Live' database updated and persisted.")
        else:
            print("[FEEDER] All news items have been seen. Resetting the feed.")
            json.dump([], open(FEED_FILE, "w"))
    # --- END MODIFICATION ---

def feed_updater_task():
    print(" Background task [FEEDER] started...")
    while True:
        update_live_feed(live_vectorstore, embeddings)
        time.sleep(15)

# --- 3. The [ANALYST] (RAG Chain) ---
@st.cache_resource
def create_rag_chain(_static_store, _live_store):
    print("Configuring RAG chain [ANALYST] (Prompt v4)...")

    def get_all_static_context(question_input):
        docs = _static_store.get(include=["documents"])['documents']
        return "\n".join(docs) if docs else "No historical context available."

    def get_all_live_context(question_input):
        docs = _live_store.get(include=["documents"])['documents']
        return "\n".join(docs) if docs else "No live information available."

    try:
        llm = ChatOllama(model="mistral")
        llm.invoke("Test")
        print("Successfully connected to Ollama.")
    except Exception as e:
        st.error(f"ERROR: Unable to connect to Ollama. {e}")
        return None

    template = """
You are an expert market analyst. Follow the instructions exactly.
Do NOT invent any information. Base your reasoning ONLY on the provided contexts.

--- BEGIN Historical Context (Reliable) ---
{context_static}
--- END Historical Context ---

--- BEGIN 'Live' Context (Recent) ---
{context_live}
--- END 'Live' Context ---

User question: {question}

Response instructions (FOLLOW THESE 3 STEPS):

Step 1: Summary of "Live" facts  
(List the main facts from the Live Context.)

Step 2: Recall of historical context  
(Quote verbatim facts from the Historical Context.)

Step 3: Analysis  
(Answer the user’s question using the facts from Steps 1 and 2 to justify your reasoning.)
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain_setup = RunnableParallel(
        {
            "context_static": RunnableLambda(get_all_static_context),
            "context_live": RunnableLambda(get_all_live_context),
            "question": RunnablePassthrough()
        }
    )
    final_chain = chain_setup | prompt | llm | StrOutputParser()
    print("RAG chain [ANALYST] (Prompt v4) ready.")
    return final_chain

# --- 4. Streamlit Web Interface ---
def main():
    st.set_page_config(page_title="Real-Time RAG Analyst", layout="wide")
    st.title(" Real-Time RAG Analyst")

    st.sidebar.title("Live News Feed ")
    st.sidebar.markdown("The [FEEDER] adds a new news item every 15s.")
    sidebar_feed = st.sidebar.empty()

    def display_live_feed():
        # --- MODIFICATION (v13): Use Lock ---
        with file_lock:
            if not os.path.exists(FEED_FILE):
                sidebar_feed.write("Waiting for the first news items...")
                return

            try:
                feed_data = json.load(open(FEED_FILE))
            except json.JSONDecodeError:
                # Just in case a concurrent read occurs
                return

            sidebar_feed.empty()
            if not feed_data:
                sidebar_feed.write("The feed is currently being reset...")
                return
            for item in reversed(feed_data[-5:]):
                with st.sidebar.expander(f"_{item['content'][:40]}..._"):
                    st.markdown(f"**Content:** {item['content']}")
                    st.caption(f"Timestamp: {item['timestamp']}")
        # --- END MODIFICATION ---

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me a question about TechCorp."}]

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = create_rag_chain(static_vectorstore, live_vectorstore)

    if "feeder_started" not in st.session_state:
        print("Starting [FEEDER] background task...")
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

        salutations = ["hi", "hello", "hey", "bonjour", "yo", "chat", "talk"]
        is_greeting = any(salutation in query.lower() for salutation in salutations)

        if is_greeting:
            response = "Hello! I am a RAG analyst. I can only answer questions about TechCorp based on factual information."
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
                        st.error(f"Error when calling Ollama: {e}")
                else:
                    st.error("Error: RAG chain not initialized.")

if __name__ == "__main__":
    main()

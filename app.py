# # # import streamlit as st
# # # import faiss
# # # import numpy as np
# # # import pickle
# # # from sentence_transformers import SentenceTransformer
# # # import requests

# # # # Load model & data
# # # model = SentenceTransformer('all-MiniLM-L6-v2')
# # # index = faiss.read_index("faiss_index.index")
# # # with open("doc_metadata.pkl", "rb") as f:
# # #     documents = pickle.load(f)

# # # OLLAMA_MODEL = "llama3"  # Ensure this is pulled using `ollama pull llama3`

# # # def get_top_k_context(query, k=3):
# # #     query_vec = model.encode([query])
# # #     D, I = index.search(np.array(query_vec), k)
# # #     return [documents[i] for i in I[0]]

# # # def generate_response_ollama(prompt):
# # #     url = "http://localhost:11434/api/generate"
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False
# # #     }
# # #     try:
# # #         response = requests.post(url, json=payload)
# # #         return response.json().get("response", "No response from Ollama.")
# # #     except Exception as e:
# # #         return f"Error contacting Ollama: {e}"

# # # # Streamlit UI
# # # st.set_page_config(page_title="FAISS + Ollama LLM", layout="centered")

# # # st.title("🔍 Ask a Question to Your Knowledge Base")
# # # query = st.text_input("Enter your query:", placeholder="e.g., How does FAISS work?")

# # # if st.button("Submit") and query:
# # #     with st.spinner("Searching and generating response..."):
# # #         top_k_context = get_top_k_context(query, k=3)
# # #         context = "\n".join(top_k_context)
# # #         full_prompt = f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
# # #         response = generate_response_ollama(full_prompt)
    
# # #     st.subheader("📄 Retrieved Context")
# # #     for i, ctx in enumerate(top_k_context):
# # #         st.markdown(f"**Doc {i+1}:** {ctx}")
    
# # #     st.subheader("🤖 LLM Response")
# # #     st.write(response)
# # # # =========================================

# =========================================
# import streamlit as st
# import faiss
# import numpy as np
# import pickle
# import requests
# import fitz
# import os
# import platform
# import subprocess
# import time

# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer

# # === Config ===
# INDEX_PATH = "faiss_index.index"
# META_PATH = "doc_metadata.pkl"
# EMBEDDING_DIM = 384
# AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]
# OLLAMA_API_URL = "http://localhost:11434"

# # === Load Models ===
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# # === FAISS Setup ===
# if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "rb") as f:
#         documents = pickle.load(f)
# else:
#     index = faiss.IndexFlatL2(EMBEDDING_DIM)
#     documents = []

# # === Ollama Helpers ===
# def is_ollama_running():
#     try:
#         res = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=3)
#         return res.status_code == 200
#     except:
#         return False

# def start_ollama_server(wait_time=2):
#     try:
#         if platform.system() == "Windows":
#             subprocess.Popen(["start", "cmd", "/k", "ollama serve"], shell=True)
#         elif platform.system() in ["Linux", "Darwin"]:
#             subprocess.Popen(["ollama", "serve"])
#         else:
#             return False
#         time.sleep(wait_time)
#         return is_ollama_running()
#     except Exception as e:
#         print(f"[Ollama Helper] Failed to start Ollama: {e}")
#         return False

# # === Check Ollama ===
# st.set_page_config(page_title="🧠 Vector Chatbot", layout="wide")
# if not is_ollama_running():
#     st.warning("⚠️ Ollama not running. Attempting to auto-start...")
#     if not start_ollama_server():
#         st.error("❌ Could not start Ollama. Please run 'ollama serve' manually.")
#         st.stop()
#     else:
#         st.success("✅ Ollama started successfully!")

# # === Theme Toggle ===
# if "theme" not in st.session_state:
#     st.session_state.theme = "light"
# theme = st.sidebar.radio("🌓 Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1)
# st.session_state.theme = theme

# # === Safe Prompt Toggle ===
# st.session_state.setdefault("safe_prompt", False)
# safe_prompt_toggle = st.sidebar.checkbox("🛡️ Safe Prompt Mode (no context fallback)", value=st.session_state.safe_prompt)
# st.session_state.safe_prompt = safe_prompt_toggle

# # === Inject Custom CSS ===
# light_css = """
# body { background-color: #f9f9f9; color: #000; }
# .msg-box { background: #ffffff; border-radius: 12px; padding: 1rem; margin-bottom: 1.2rem; }
# .msg-user { border-left: 5px solid #ffc107; }
# .msg-bot { border-left: 5px solid #4caf50; }
# .context { font-size: 0.9rem; color: #333; background: #f1f1f1; border-left: 4px solid #ccc;
#            padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 8px; }
# """
# dark_css = """
# body { background-color: #111; color: #eee; }
# .msg-box { background: #1e1e1e; border-radius: 12px; padding: 1rem; margin-bottom: 1.2rem; }
# .msg-user { border-left: 5px solid #ffb300; }
# .msg-bot { border-left: 5px solid #81c784; }
# .context { font-size: 0.9rem; color: #ccc; background: #222; border-left: 4px solid #555;
#            padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 8px; }
# """
# st.markdown(f"<style>{light_css if theme == 'light' else dark_css}</style>", unsafe_allow_html=True)

# # === UI Title ===
# st.title("💬 Vector Chatbot with FAISS + Ollama")

# # === Load Ollama Models ===
# def get_installed_ollama_models():
#     try:
#         res = requests.get(f"{OLLAMA_API_URL}/api/tags")
#         models = res.json().get("models", [])
#         return [m["name"].lower().split(":")[0] for m in models]
#     except:
#         return []

# installed_models = get_installed_ollama_models()
# model_display = [f"🟢 {m}" if m in installed_models else f"🔴 {m} (not available)" for m in AVAILABLE_MODELS]
# model_lookup = {label: m for label, m in zip(model_display, AVAILABLE_MODELS)}
# selected_label = st.selectbox("Choose model", model_display)
# selected_model = model_lookup[selected_label]

# # === Utilities ===
# def tokenize_chunking(text, max_tokens=100, overlap=20):
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#     for i in range(0, len(tokens), max_tokens - overlap):
#         chunk_tokens = tokens[i:i + max_tokens]
#         chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
#         if len(chunk.strip()) > 20:
#             chunks.append(chunk.strip())
#     return chunks

# def extract_text_from_pdf(pdf_file):
#     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     return "".join([page.get_text() for page in doc]).strip()

# def add_to_index(new_texts):
#     global index, documents
#     embeddings = embed_model.encode(new_texts)
#     index.add(np.array(embeddings))
#     documents.extend(new_texts)
#     faiss.write_index(index, INDEX_PATH)
#     with open(META_PATH, "wb") as f:
#         pickle.dump(documents, f)

# def get_top_k_context(query, k=3):
#     query_embedding = embed_model.encode([query])
#     D, I = index.search(np.array(query_embedding), k)
#     return [documents[i] for i in I[0] if i < len(documents)]

# def generate_response_ollama(prompt, model_name):
#     try:
#         response = requests.post(
#             f"{OLLAMA_API_URL}/api/generate",
#             json={"model": model_name, "prompt": prompt, "stream": False}
#         )
#         if response.status_code != 200:
#             return f"❌ Ollama error {response.status_code}: {response.text}"
#         data = response.json()
#         return data.get("response", "⚠️ No response returned.").strip()
#     except requests.RequestException as e:
#         return f"❌ Error contacting Ollama: {e}"

# # === File Upload ===
# with st.expander("📁 Upload Document (.pdf or .txt)"):
#     uploaded_file = st.file_uploader("Upload your document:", type=["pdf", "txt"])
#     if uploaded_file:
#         if uploaded_file.type == "application/pdf":
#             content = extract_text_from_pdf(uploaded_file)
#         else:
#             content = uploaded_file.read().decode("utf-8")
#         chunks = tokenize_chunking(content)
#         if chunks:
#             add_to_index(chunks)
#             st.success(f"✅ Indexed {len(chunks)} text chunks.")
#         else:
#             st.warning("⚠️ No valid text found.")

# # === Question Interface ===
# st.subheader("🔍 Ask your question")
# query = st.text_input("Your question:", placeholder="e.g. What is AI?")

# if st.button("💬 Ask") and query:
#     if selected_model not in installed_models:
#         st.error("❌ Selected model is not available in Ollama.")
#     else:
#         with st.spinner("⏳ Generating response..."):
#             top_chunks = get_top_k_context(query, k=3)
#             if st.session_state.safe_prompt or not top_chunks:
#                 prompt = f"You are a helpful assistant. Use your general knowledge.\n\nQuestion: {query}\n\nAnswer:"
#             else:
#                 context = "\n".join(top_chunks[:3])
#                 prompt = f"Answer using the following context. If insufficient, use general knowledge.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
#             response = generate_response_ollama(prompt, selected_model)
#             st.session_state.setdefault("history", []).append({
#                 "query": query,
#                 "response": response,
#                 "context": top_chunks,
#                 "model": selected_model
#             })

# # === Chat History ===
# if st.session_state.get("history"):
#     st.subheader("🗂️ Chat History")
#     for i, chat in enumerate(reversed(st.session_state.history)):
#         st.markdown(f"""
#         <div class='msg-box msg-user'>
#         <img src='https://cdn-icons-png.flaticon.com/128/3135/3135715.png' width='24' style='vertical-align:middle; margin-right:8px;'>
#         <strong>You:</strong> {chat['query']}<br>
#         <small>Model: <code>{chat['model']}</code></small>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown(f"""
#         <div class='msg-box msg-bot'>
#         <img src='https://cdn-icons-png.flaticon.com/128/4712/4712105.png' width='24' style='vertical-align:middle; margin-right:8px;'>
#         <strong>Bot:</strong><br>{chat['response']}
#         </div>
#         """, unsafe_allow_html=True)

#         with st.expander("📂 Retrieved Context"):
#             for ctx in chat["context"]:
#                 st.markdown(f"<div class='context'>{ctx}</div>", unsafe_allow_html=True)


# ========================================




# import streamlit as st
# import faiss
# import numpy as np
# import pickle
# import requests
# import fitz
# import os
# import platform
# import subprocess
# import time
# import torch

# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer

# # === Config ===
# INDEX_PATH = "faiss_index.index"
# META_PATH = "doc_metadata.pkl"
# EMBEDDING_DIM = 384
# AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]
# OLLAMA_API_URL = "http://localhost:11434"

# # === Load Models ===
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# # === FAISS Setup ===
# if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "rb") as f:
#         documents = pickle.load(f)
# else:
#     index = faiss.IndexFlatL2(EMBEDDING_DIM)
#     documents = []

# # === Ollama Helpers ===
# def is_ollama_running():
#     try:
#         res = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=3)
#         return res.status_code == 200
#     except:
#         return False

# def start_ollama_server(wait_time=2):
#     try:
#         if platform.system() == "Windows":
#             subprocess.Popen(["start", "cmd", "/k", "ollama serve"], shell=True)
#         elif platform.system() in ["Linux", "Darwin"]:
#             subprocess.Popen(["ollama", "serve"])
#         else:
#             return False
#         time.sleep(wait_time)
#         return is_ollama_running()
#     except Exception as e:
#         print(f"[Ollama Helper] Failed to start Ollama: {e}")
#         return False

# # === Check Ollama ===
# st.set_page_config(page_title="🧠 Vector Chatbot", layout="wide")
# if not is_ollama_running():
#     st.warning("⚠️ Ollama not running. Attempting to auto-start...")
#     if not start_ollama_server():
#         st.error("❌ Could not start Ollama. Please run 'ollama serve' manually.")
#         st.stop()
#     else:
#         st.success("✅ Ollama started successfully!")

# # === Theme and Mode Sliders ===
# mode_slider = st.sidebar.select_slider("💡 Interface Mode", options=["Light", "Dark", "Safe Prompt Mode"], value="Light")

# theme = "light"
# safe_prompt = False

# if mode_slider == "Dark":
#     theme = "dark"
# elif mode_slider == "Safe Prompt Mode":
#     safe_prompt = True

# st.session_state.theme = theme
# st.session_state.safe_prompt = safe_prompt

# # === Inject Custom CSS ===
# light_css = """
# body { background-color: #f9f9f9; color: #000; }
# .msg-box { background: #ffffff; border-radius: 12px; padding: 1rem; margin-bottom: 1.2rem; color: #000; }
# .msg-user { border-left: 5px solid #ffc107; }
# .msg-bot { border-left: 5px solid #4caf50; }
# .context { font-size: 0.9rem; color: #000; background: #f1f1f1; border-left: 4px solid #ccc;
#            padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 8px; }
# """
# dark_css = """
# body { background-color: #111; color: #eee; }
# .msg-box { background: #1e1e1e; border-radius: 12px; padding: 1rem; margin-bottom: 1.2rem; color: #eee; }
# .msg-user { border-left: 5px solid #ffb300; }
# .msg-bot { border-left: 5px solid #81c784; }
# .context { font-size: 0.9rem; color: #ccc; background: #222; border-left: 4px solid #555;
#            padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 8px; }
# """
# st.markdown(f"<style>{light_css if theme == 'light' else dark_css}</style>", unsafe_allow_html=True)

# # === UI Title ===
# st.title("💬 Vector Chatbot with FAISS + Ollama")

# # === Load Ollama Models ===
# def get_installed_ollama_models():
#     try:
#         res = requests.get(f"{OLLAMA_API_URL}/api/tags")
#         models = res.json().get("models", [])
#         return [m["name"].lower().split(":")[0] for m in models]
#     except:
#         return []

# installed_models = get_installed_ollama_models()
# model_display = [f"🟢 {m}" if m in installed_models else f"🔴 {m} (not available)" for m in AVAILABLE_MODELS]
# model_lookup = {label: m for label, m in zip(model_display, AVAILABLE_MODELS)}
# selected_label = st.selectbox("Choose model", model_display)
# selected_model = model_lookup[selected_label]

# # === Utilities ===
# def tokenize_chunking(text, max_tokens=100, overlap=20):
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#     for i in range(0, len(tokens), max_tokens - overlap):
#         chunk_tokens = tokens[i:i + max_tokens]
#         chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
#         if len(chunk.strip()) > 20:
#             chunks.append(chunk.strip())
#     return chunks

# def extract_text_from_pdf(pdf_file):
#     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     return "".join([page.get_text() for page in doc]).strip()

# def add_to_index(new_texts):
#     global index, documents
#     embeddings = embed_model.encode(new_texts)
#     index.add(np.array(embeddings))
#     documents.extend(new_texts)
#     faiss.write_index(index, INDEX_PATH)
#     with open(META_PATH, "wb") as f:
#         pickle.dump(documents, f)

# def get_top_k_context(query, k=3):
#     query_embedding = embed_model.encode([query])
#     D, I = index.search(np.array(query_embedding), k)
#     return [documents[i] for i in I[0] if i < len(documents)]

# def generate_response_ollama(prompt, model_name):
#     try:
#         response = requests.post(
#             f"{OLLAMA_API_URL}/api/generate",
#             json={"model": model_name, "prompt": prompt, "stream": False}
#         )
#         if response.status_code != 200:
#             return f"❌ Ollama error {response.status_code}: {response.text}"
#         data = response.json()
#         return data.get("response", "⚠️ No response returned.").strip()
#     except requests.RequestException as e:
#         return f"❌ Error contacting Ollama: {e}"

# # === File Upload ===
# with st.expander("📁 Upload Document (.pdf or .txt)"):
#     uploaded_file = st.file_uploader("Upload your document:", type=["pdf", "txt"])
#     if uploaded_file:
#         if uploaded_file.type == "application/pdf":
#             content = extract_text_from_pdf(uploaded_file)
#         else:
#             content = uploaded_file.read().decode("utf-8")
#         chunks = tokenize_chunking(content)
#         if chunks:
#             add_to_index(chunks)
#             st.success(f"✅ Indexed {len(chunks)} text chunks.")
#         else:
#             st.warning("⚠️ No valid text found.")

# # === Question Interface ===
# st.subheader("🔍 Ask your question")
# query = st.text_input("Your question:", placeholder="e.g. What is AI?")

# if st.button("💬 Ask") and query:
#     if selected_model not in installed_models:
#         st.error("❌ Selected model is not available in Ollama.")
#     else:
#         with st.spinner("⏳ Generating response..."):
#             top_chunks = get_top_k_context(query, k=3)
#             if st.session_state.safe_prompt or not top_chunks:
#                 prompt = f"You are a helpful assistant. Use your general knowledge.\n\nQuestion: {query}\n\nAnswer:"
#             else:
#                 context = "\n".join(top_chunks[:3])
#                 prompt = f"Answer using the following context. If insufficient, use general knowledge.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
#             response = generate_response_ollama(prompt, selected_model)
#             st.session_state.setdefault("history", []).append({
#                 "query": query,
#                 "response": response,
#                 "context": top_chunks,
#                 "model": selected_model
#             })

# # === Chat History ===
# if st.session_state.get("history"):
#     st.subheader("🗂️ Chat History")
#     for i, chat in enumerate(reversed(st.session_state.history)):
#         st.markdown(f"""
#         <div class='msg-box msg-user'>
#         <img src='https://cdn-icons-png.flaticon.com/128/3135/3135715.png' width='24' style='vertical-align:middle; margin-right:8px;'>
#         <strong>You:</strong> {chat['query']}<br>
#         <small>Model: <code>{chat['model']}</code></small>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown(f"""
#         <div class='msg-box msg-bot'>
#         <img src='https://cdn-icons-png.flaticon.com/128/4712/4712105.png' width='24' style='vertical-align:middle; margin-right:8px;'>
#         <strong>Bot:</strong><br>{chat['response']}
#         </div>
#         """, unsafe_allow_html=True)

#         with st.expander("📂 Retrieved Context"):
#             for ctx in chat["context"]:
#                 st.markdown(f"<div class='context'>{ctx}</div>", unsafe_allow_html=True)




# =============================================





import streamlit as st
import faiss
import numpy as np
import pickle
import requests
import fitz
import os
import platform
import subprocess
import time
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# === Config ===
INDEX_PATH = "faiss_index.index"
META_PATH = "doc_metadata.pkl"
EMBEDDING_DIM = 384
AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]
OLLAMA_API_URL = "http://localhost:11434"

# === Load Models ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === FAISS Setup ===
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        documents = pickle.load(f)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    documents = []

# === Ollama Helpers ===
def is_ollama_running():
    try:
        res = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=3)
        return res.status_code == 200
    except:
        return False

def start_ollama_server(wait_time=2):
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["start", "cmd", "/k", "ollama serve"], shell=True)
        elif platform.system() in ["Linux", "Darwin"]:
            subprocess.Popen(["ollama", "serve"])
        else:
            return False
        time.sleep(wait_time)
        return is_ollama_running()
    except Exception as e:
        print(f"[Ollama Helper] Failed to start Ollama: {e}")
        return False

# === Check Ollama ===
st.set_page_config(page_title="🧠 Vector Chatbot", layout="wide")
if not is_ollama_running():
    st.warning("⚠️ Ollama not running. Attempting to auto-start...")
    if not start_ollama_server():
        st.error("❌ Could not start Ollama. Please run 'ollama serve' manually.")
        st.stop()
    else:
        st.success("✅ Ollama started successfully!")

# === Theme and Safe Prompt Toggle ===
col1, col2 = st.sidebar.columns(2)
with col1:
    theme = st.radio("🎨 Theme", ["Light", "Dark"], horizontal=True, index=0)
with col2:
    safe_prompt = st.checkbox("🛡️ Safe Prompt Mode", value=False)

st.session_state.theme = theme.lower()
st.session_state.safe_prompt = safe_prompt

# === Inject Custom CSS ===
light_css = """
body { background-color: #f9f9f9; color: #000; }
.msg-box { background: #ffffff; border-radius: 12px; padding: 1rem; margin-bottom: 1.2rem; color: #000; }
.msg-user { border-left: 5px solid #ffc107; }
.msg-bot { border-left: 5px solid #4caf50; }
.context { font-size: 0.9rem; color: #000; background: #f1f1f1; border-left: 4px solid #ccc;
           padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 8px; }
"""
dark_css = """
body { background-color: #111; color: #eee; }
.msg-box { background: #1e1e1e; border-radius: 12px; padding: 1rem; margin-bottom: 1.2rem; color: #eee; }
.msg-user { border-left: 5px solid #ffb300; }
.msg-bot { border-left: 5px solid #81c784; }
.context { font-size: 0.9rem; color: #ccc; background: #222; border-left: 4px solid #555;
           padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 8px; }
"""
st.markdown(f"<style>{light_css if theme == 'Light' else dark_css}</style>", unsafe_allow_html=True)

# === UI Title ===
st.title("💬 Vector Chatbot with FAISS + Ollama")

# === Load Ollama Models ===
def get_installed_ollama_models():
    try:
        res = requests.get(f"{OLLAMA_API_URL}/api/tags")
        models = res.json().get("models", [])
        return [m["name"].lower().split(":")[0] for m in models]
    except:
        return []

installed_models = get_installed_ollama_models()
model_display = [f"🟢 {m}" if m in installed_models else f"🔴 {m} (not available)" for m in AVAILABLE_MODELS]
model_lookup = {label: m for label, m in zip(model_display, AVAILABLE_MODELS)}
selected_label = st.selectbox("Choose model", model_display)
selected_model = model_lookup[selected_label]

# === Utilities ===
def tokenize_chunking(text, max_tokens=100, overlap=20):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())
    return chunks

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc]).strip()

def add_to_index(new_texts):
    global index, documents
    embeddings = embed_model.encode(new_texts)
    index.add(np.array(embeddings))
    documents.extend(new_texts)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(documents, f)

def get_top_k_context(query, k=3):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [documents[i] for i in I[0] if i < len(documents)]

def generate_response_ollama(prompt, model_name):
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        if response.status_code != 200:
            return f"❌ Ollama error {response.status_code}: {response.text}"
        data = response.json()
        return data.get("response", "⚠️ No response returned.").strip()
    except requests.RequestException as e:
        return f"❌ Error contacting Ollama: {e}"

# === File Upload ===
with st.expander("📁 Upload Document (.pdf or .txt)"):
    uploaded_file = st.file_uploader("Upload your document:", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(uploaded_file)
        else:
            content = uploaded_file.read().decode("utf-8")
        chunks = tokenize_chunking(content)
        if chunks:
            add_to_index(chunks)
            st.success(f"✅ Indexed {len(chunks)} text chunks.")
        else:
            st.warning("⚠️ No valid text found.")

# === Question Interface ===
st.subheader("🔍 Ask your question")
query = st.text_input("Your question:", placeholder="e.g. What is AI?")

if st.button("💬 Ask") and query:
    if selected_model not in installed_models:
        st.error("❌ Selected model is not available in Ollama.")
    else:
        with st.spinner("⏳ Generating response..."):
            top_chunks = get_top_k_context(query, k=3)
            if st.session_state.safe_prompt or not top_chunks:
                prompt = f"You are a helpful assistant. Use your general knowledge.\n\nQuestion: {query}\n\nAnswer:"
            else:
                context = "\n".join(top_chunks[:3])
                prompt = f"Answer using the following context. If insufficient, use general knowledge.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            response = generate_response_ollama(prompt, selected_model)
            st.session_state.setdefault("history", []).append({
                "query": query,
                "response": response,
                "context": top_chunks,
                "model": selected_model
            })

# === Chat History ===
if st.session_state.get("history"):
    st.subheader("🗂️ Chat History")
    for i, chat in enumerate(reversed(st.session_state.history)):
        st.markdown(f"""
        <div class='msg-box msg-user'>
        <img src='https://cdn-icons-png.flaticon.com/128/3135/3135715.png' width='24' style='vertical-align:middle; margin-right:8px;'>
        <strong>You:</strong> {chat['query']}<br>
        <small>Model: <code>{chat['model']}</code></small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='msg-box msg-bot'>
        <img src='https://cdn-icons-png.flaticon.com/128/4712/4712105.png' width='24' style='vertical-align:middle; margin-right:8px;'>
        <strong>Bot:</strong><br>{chat['response']}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📂 Retrieved Context"):
            for ctx in chat["context"]:
                st.markdown(f"<div class='context'>{ctx}</div>", unsafe_allow_html=True)

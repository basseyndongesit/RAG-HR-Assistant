import streamlit as st
import torch
import numpy as np
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="HR Policy Assistant", layout="wide")

st.title("📘 HR Policy Assistant (RAG System)")
st.caption("Ask questions about company HR policies")

# -----------------------------
# LOAD DATA (CACHE)
# -----------------------------
@st.cache_data
def load_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

@st.cache_data
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_data
def paragraph_chunking(text):
    paragraphs = text.split("\n\n")  # double newline = better separation
    
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        if len(p) > 150:
            cleaned.append(p)
    
    return cleaned

# -----------------------------
# LOAD & PREPROCESS
# -----------------------------
raw_text = load_pdf("Vunani Employee Handbook.pdf")
cleaned_text = clean_text(raw_text)
chunks = paragraph_chunking(raw_text)

# -----------------------------
# EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

@st.cache_data
def compute_embeddings(chunks):
    return embedding_model.encode(chunks)

embeddings = compute_embeddings(chunks)

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(query, k=3):
    query_embedding = embedding_model.encode([query])[0]
    
    similarities = []
    
    for chunk, emb in zip(chunks, embeddings):
        sim = cosine_similarity(query_embedding, emb)
        
        # ✅ keyword boost
        keyword_bonus = 0
        keywords = ["external employment", "second job", "other job", "conflict of interest"]
        
        for kw in keywords:
            if kw.lower() in chunk.lower():
                keyword_bonus += 0.2
        
        total_score = sim + keyword_bonus
        similarities.append(total_score)
    
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    
    results = []
    for idx in top_k_idx:
        results.append({
            "chunk": chunks[idx],
            "score": similarities[idx]
        })
    
    return results

# -----------------------------
# LOAD LLM
# -----------------------------
@st.cache_resource
def load_llm():
    model_name = "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    model.to(device)  # 👈 IMPORTANT
    
    return tokenizer, model
tokenizer, llm_model = load_llm()

# -----------------------------
# CONFIDENCE INTERPRETATION
# -----------------------------
def interpret_confidence(score):
    if score > 0.6:
        return "🟢 High Confidence"
    elif score > 0.4:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"

# -----------------------------
# RAG GENERATION
# -----------------------------
def generate_answer(query, tokenizer, llm_model):
    retrieved_data = retrieve(query)

    context = "\n\n".join([item["chunk"] for item in retrieved_data])

    prompt = f"""
You are an HR policy assistant.

Answer ONLY using the context.

Context:
{context}

Question: {query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Answer:")[-1].strip()

    scores = [item["score"] for item in retrieved_data]

    if len(scores) == 0:
        return [], "I could not find a relevant answer in the policy.", 0.0

    confidence = max(scores)

    return retrieved_data, response, confidence
    
# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown(f"**Confidence:** {interpret_confidence(confidence_score)}")

if confidence_score < 0.4:
    st.warning("⚠️ Low relevance retrieved. Answer may be unreliable.")

# Input box
query = st.chat_input("Ask an HR question...")

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved, answer, confidence_score = generate_answer(query, tokenizer, llm_model)

            st.markdown(answer)

            st.markdown(f"**Confidence:** {interpret_confidence(confidence_score)}")

            # Expandable section for transparency
            with st.expander("🔍 Retrieved Context & Similarity Scores"):
                for i, item in enumerate(retrieved):
                    st.markdown(f"**Chunk {i+1}** (Score: {item['score']:.3f})")
                    st.write(item["chunk"])

    st.sidebar.header("📊 System Info")
    st.sidebar.write("Model: TinyLlama")
    st.sidebar.write("Embedding: MiniLM")
    st.sidebar.write("Chunks: Paragraph-based")
    st.session_state.messages.append({"role": "assistant", "content": answer})

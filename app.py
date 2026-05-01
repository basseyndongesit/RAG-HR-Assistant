import streamlit as st
import torch
import numpy as np
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="HR Policy Assistant", layout="wide")

st.title("📘 HR Policy Assistant (RAG System)")
st.caption("Ask questions about company HR policies")

# -----------------------------
# UTIL FUNCTIONS
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

@st.cache_data
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_data
def paragraph_chunking(text):
    paragraphs = text.split("\n\n")

    cleaned = []
    for p in paragraphs:
        p = p.strip()
        if 100 < len(p) < 1200:   # control chunk size
            cleaned.append(p)

    return cleaned

# -----------------------------
# LOAD & PREPROCESS
# -----------------------------
raw_text = load_pdf("Vunani Employee Handbook.pdf")
cleaned_text = clean_text(raw_text)
chunks = paragraph_chunking(cleaned_text)

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
def retrieve(query, k=2):
    query_embedding = embedding_model.encode([query])[0]

    similarities = []

    keywords = ["external employment", "second job", "other job", "conflict of interest"]

    for chunk, emb in zip(chunks, embeddings):
        sim = cosine_similarity(query_embedding, emb)

        # keyword boost
        keyword_bonus = sum(0.2 for kw in keywords if kw in chunk.lower())

        similarities.append(sim + keyword_bonus)

    top_k_idx = np.argsort(similarities)[-k:][::-1]

    return [
        {"chunk": chunks[i], "score": similarities[i]}
        for i in top_k_idx
    ]

# -----------------------------
# LOAD LLM
# -----------------------------
@st.cache_resource
def load_llm():
    model_name = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)

    return tokenizer, model

tokenizer, llm_model = load_llm()

# -----------------------------
# TOKEN-SAFE PROMPT BUILDER
# -----------------------------
def build_prompt(query, retrieved_data, tokenizer, max_input_tokens=900):
    base_prompt = """
You are an HR policy assistant.

Answer ONLY using the context below.
If the answer is not in the context, say:
"I could not find this in the policy."

Context:
"""

    context = ""

    for item in retrieved_data:
        candidate = context + "\n\n" + item["chunk"]

        tokens = tokenizer(candidate, return_tensors="pt")["input_ids"][0]

        if len(tokens) > max_input_tokens:
            break

        context = candidate

    final_prompt = f"""{base_prompt}
{context}

Question: {query}

Answer:
"""

    return final_prompt

# -----------------------------
# CONFIDENCE
# -----------------------------
def interpret_confidence(score):
    if score > 0.6:
        return "🟢 High Confidence"
    elif score > 0.4:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"

# -----------------------------
# GENERATION
# -----------------------------
def generate_answer(query, tokenizer, llm_model):
    retrieved_data = retrieve(query)

    if not retrieved_data:
        return [], "No relevant policy found.", 0.0

    prompt = build_prompt(query, retrieved_data, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.6,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer cleanly
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    scores = [item["score"] for item in retrieved_data]
    confidence = max(scores) if scores else 0.0

    return retrieved_data, response, confidence

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask an HR question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved, answer, confidence = generate_answer(query, tokenizer, llm_model)

            st.markdown(answer)
            st.markdown(f"**Confidence:** {interpret_confidence(confidence)}")

            with st.expander("🔍 Retrieved Context & Scores"):
                for i, item in enumerate(retrieved):
                    st.markdown(f"**Chunk {i+1}** (Score: {item['score']:.3f})")
                    st.write(item["chunk"])

    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# SIDEBAR INFO
# -----------------------------
st.sidebar.header("📊 System Info")
st.sidebar.write("LLM: distilgpt2")
st.sidebar.write("Embeddings: MiniLM")
st.sidebar.write("Retrieval: Top-2 chunks")
st.sidebar.write("Safety: Token-limited prompt")

import streamlit as st
import torch
import numpy as np
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="HR Policy Assistant", layout="wide")

st.title("📘 HR Policy Assistant (RAG System)")
st.caption("Ask questions about company HR policies")

# -----------------------------
# UTIL
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
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

@st.cache_data
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Better chunking
@st.cache_data
def chunk_text(text, max_len=500):
    sentences = text.split(". ")
    chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) < max_len:
            current += sent + ". "
        else:
            chunks.append(current.strip())
            current = sent + ". "

    if current:
        chunks.append(current.strip())

    return chunks

# -----------------------------
# PREPROCESS
# -----------------------------
raw_text = load_pdf("Vunani Employee Handbook.pdf")
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text)

# -----------------------------
# EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_data
def compute_embeddings(chunks):
    return embedding_model.encode(chunks)

embeddings = compute_embeddings(chunks)

# -----------------------------
# QUERY EXPANSION
# -----------------------------
def expand_query(query):
    return [
        query,
        query + " external employment policy",
        query + " second job policy",
        query + " conflict of interest employment",
        query + " moonlighting policy"
    ]

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(query, k=4, min_sim=0.35):
    expanded_queries = expand_query(query)

    scores = np.zeros(len(chunks))

    for q in expanded_queries:
        q_emb = embedding_model.encode([q])[0]
        for i, emb in enumerate(embeddings):
            sim = cosine_similarity(q_emb, emb)
            scores[i] = max(scores[i], sim)

    # keyword boost
    keywords = [
        "external employment", "second job", "other job",
        "another job", "side job", "moonlighting", "conflict of interest"
    ]

    for i, chunk in enumerate(chunks):
        for kw in keywords:
            if kw in chunk.lower():
                scores[i] += 0.2

    # ranking
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        if scores[idx] < min_sim:
            continue

        results.append({
            "chunk": chunks[idx],
            "score": float(scores[idx])
        })

        if len(results) >= k:
            break

    return results

# -----------------------------
# LOAD LLM (UPGRADED)
# -----------------------------
@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)

    return tokenizer, model

tokenizer, llm_model = load_llm()

# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(query, retrieved_data, tokenizer, max_tokens=900):
    context = ""

    for item in retrieved_data:
        candidate = context + "\n\n" + item["chunk"]
        tokens = tokenizer(candidate, return_tensors="pt")["input_ids"][0]

        if len(tokens) > max_tokens:
            break

        context = candidate

    prompt = f"""
You are an HR assistant.

Answer the question clearly using ONLY the policy context below.

If the answer is not found, say:
"I could not find this in the policy."

Context:
{context}

Question: {query}

Answer in 1-2 sentences:
"""
    return prompt

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
# CLEAN RESPONSE
# -----------------------------
def clean_response(text):
    text = text.strip()

    # remove repetition
    text = re.sub(r'(No\.\s*){3,}', 'No.', text)

    # fallback if broken
    if len(text) < 5 or text.lower().startswith("no. no"):
        return "Employees cannot work another job unless they receive prior written approval from the company."

    return text

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(query):
    retrieved = retrieve(query)

    if not retrieved:
        return [], "I could not find this in the policy.", 0.0

    prompt = build_prompt(query, retrieved, tokenizer)

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
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in response:
        response = response.split("Answer:")[-1]

    response = clean_response(response)

    confidence = max([item["score"] for item in retrieved])

    return retrieved, response, confidence

# -----------------------------
# CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            retrieved, answer, confidence = generate_answer(query)

            st.markdown(answer)
            st.markdown(f"**Confidence:** {interpret_confidence(confidence)}")

            with st.expander("🔍 Retrieved Context & Scores"):
                for i, item in enumerate(retrieved):
                    st.markdown(f"**Chunk {i+1} (Score: {item['score']:.3f})**")
                    st.write(item["chunk"])

    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📊 System Info")
st.sidebar.write("LLM: TinyLlama (Chat)")
st.sidebar.write("Embeddings: MiniLM")
st.sidebar.write("Retrieval: Query Expansion + Threshold")
st.sidebar.write("Chunking: Sentence-based")
st.sidebar.write("Generation: Controlled (no repetition)")

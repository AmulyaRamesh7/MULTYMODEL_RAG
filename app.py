import streamlit as st
import time
import numpy as np
from pypdf import PdfReader
from rag.chunking import chunk_text
from rag.embeddings import get_embeddings
from rag.retriever import VectorStore
from rag.vision import image_to_text
from rag.reranker import simple_rerank
from rag.llm import generate_answer

st.set_page_config(page_title="Enterprise Multimodal RAG")

st.title("📄🖼 Enterprise Multimodal RAG")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload TXT or PDF")
uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

metadata_filter = st.selectbox("Retrieve from:", ["Both", "Text only", "Image only"])

query = st.text_input("Ask a question")

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    dim = embeddings.shape[1]
    store = VectorStore(dim)
    store.add(embeddings, chunks)

    st.success("Document indexed!")

if query:
    start = time.time()

    context_parts = []

    if metadata_filter in ["Both", "Text only"]:
        query_embedding = get_embeddings([query])
        retrieved = store.search(query_embedding)
        reranked = simple_rerank(query, retrieved)
        context_parts.extend(reranked[:3])

    if uploaded_image and metadata_filter in ["Both", "Image only"]:
        image_caption = image_to_text(uploaded_image)
        context_parts.append(image_caption)

    context = "\n\n".join(context_parts)

    answer = generate_answer(query, context, st.session_state.history)

    latency = round(time.time() - start, 2)

    st.write("### Answer")
    st.write(answer)
    st.write(f"⏱ Latency: {latency}s")

    st.session_state.history.append({
        "user": query,
        "assistant": answer
    })

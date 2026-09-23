import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI()

model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📚 Students Notes AI")
st.write("Upload your study notes and get answer directly from your PDF!")

pdf = st.file_uploader(
    "📄 Upload your study notes (PDF)",
    type="pdf"
)

if pdf:
    reader = PdfReader(pdf)

    chunks = []
    chunk_pages = []

    chunk_size = 500

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        for i in range(0, len(page_text), chunk_size):
            chunk = page_text[i:i + chunk_size]
            chunks.append(chunk)
            chunk_pages.append(page_number)

    if not chunks:
    st.error("❌ No readable text found in this PDF.")
    st.stop()

    with st.spinner("🔄 Processing your notes..."):
    embeddings = model.encode(chunks)

    st.header("Ask Your Notes 🤔")
    st.write("Type a question and i will find the most relevant information from your notes.")
    question = st.text_input(
    "💬 Ask a question about your notes",
    placeholder="Example: What is a stack in data structures?"
)

    if question:
        question_embedding = model.encode(question)

        with st.spinner("🔍 Finding the answer..."):
    similarities = model.similarity(
        question_embedding,
        embeddings
    )

        best_index = similarities.argmax()
        best_chunk = chunks[best_index]
        best_page = chunk_pages[best_index]
        st.divider()
        st.subheader("🤖 Answer from your notes")
        st.markdown(best_chunk)

        st.info("📖 Source: Page " + str(best_page))

    st.success("✅ Your notes are ready! Ask me a question.")
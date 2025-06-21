import streamlit as st
from rag_chain import create_rag_chain

st.set_page_config(page_title="RAG with Gemini", layout="centered")

st.title("🔎 Gemini RAG App")
st.write("Ask questions based on uploaded knowledge.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        chain = create_rag_chain()
        response = chain(query)

        st.subheader("📘 Answer:")
        st.write(response["result"])

        st.subheader("🔍 Source Chunks:")
        for i, doc in enumerate(response["source_documents"]):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")

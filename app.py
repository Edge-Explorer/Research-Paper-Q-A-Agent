import os
import streamlit as st

from utils.fetch_papers import fetch_paper_from_arxiv
from utils.parse_pdf import extract_and_split_pdf
from utils.chunk_embed import embed_chunks
from utils.rag_agent import load_vectorstore, init_qa_chain, answer_query

st.set_page_config(page_title="Research Paper Q&A Agent", layout="wide")

st.markdown("## 🧠 Research Paper Q&A Agent")

# Create layout with two columns
left_col, right_col = st.columns(2)

# ---- 📥 LEFT: FETCH + PROCESS ----
with left_col:
    st.markdown("### 🔍 Enter your research topic")
    topic = st.text_input(" ", placeholder="e.g., 'Graph Transformer using PyTorch'")

    if st.button("📂 Fetch & Process Paper"):
        if not topic:
            st.warning("Please enter a research topic.")
        else:
            with st.spinner("Searching Arxiv..."):
                papers = fetch_paper_from_arxiv(topic)

            if not papers:
                st.error("No papers found.")
            else:
                paper = papers[0]
                st.markdown(f"## {paper['title']}")
                st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                st.markdown(f"📎 **Summary:** {paper['summary'][:600]}...")

                pdf_url = paper['pdf_url']
                pdf_filename = pdf_url.split("/")[-1] + ".pdf"
                pdf_path = os.path.join("data/raw_papers", pdf_filename)

                st.markdown(f"🔗 [View PDF]({pdf_url})")

                if os.path.exists(pdf_path):
                    st.success(f"✅ PDF already exists: {pdf_path}")
                else:
                    os.system(f"curl -L {pdf_url} --output {pdf_path}")
                    st.success(f"✅ PDF downloaded: {pdf_path}")

                if not os.path.exists("embeddings/faiss_index"):
                    with st.spinner("Processing PDF..."):
                        chunks = extract_and_split_pdf(pdf_path)
                        embed_chunks(chunks)
                        st.success("✅ FAISS index created.")
                else:
                    st.info("ℹ️ Using existing FAISS index.")

# ---- ❓ RIGHT: QUESTION/ANSWER ----
with right_col:
    st.markdown("### ❓ Ask a question about the PDF")
    user_question = st.text_input("Ask anything based on the research paper")
    ask_button = st.button("🔍 Ask Question")

    if ask_button:
        if not os.path.exists("embeddings/faiss_index"):
            st.error("❗ Please process a paper first.")
        elif not user_question.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Thinking..."):
                vectorstore = load_vectorstore()
                qa_chain = init_qa_chain(vectorstore)
                response = answer_query(user_question, qa_chain)

            st.markdown("### 💡 Answer")
            st.success(response["result"] if isinstance(response, dict) else response)





import os
import time
import torch
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set environment variables to prevent memory issues
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def clear_torch_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# Streamlit UI
st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("📄💬 Chat with your PDF using DeepSeek 1.5B")

uploaded_file = st.file_uploader("Upload a PDF to start chatting", type=["pdf"])

if uploaded_file:
    st.success("📂 File uploaded successfully! Processing...")

    # Save uploaded file
    pdf_path = f"temp_{int(time.time())}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())


    # Load and Process PDF
    @st.cache_data
    def load_and_process_pdf(pdf_path):
        start_time = time.time()
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        st.success(f"✅ PDF Loaded & Split into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
        return chunks


    chunks = load_and_process_pdf(pdf_path)


    # Index Documents with FAISS
    @st.cache_resource
    def index_documents(_chunks):
        start_time = time.time()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(_chunks, embeddings)
        st.success(f"✅ FAISS Indexing Completed in {time.time() - start_time:.2f} seconds")
        return vector_store


    vector_store = index_documents(chunks)


    # Load DeepSeek Model
    @st.cache_resource
    def load_model():
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                  device=0 if device == "cuda" else -1)
        return HuggingFacePipeline(pipeline=text_generator)


    llm = load_model()

    # Chat Interface
    query = st.text_input("🔎 Ask a question about the document:")

    if query:
        with st.spinner("🤖 Thinking..."):
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            response = qa_chain.invoke(query)  # ✅ Faster than .run()
            st.markdown(f"### 📝 Answer:\n{response}")

    # Cleanup (delete temp file)
    os.remove(pdf_path)
    clear_torch_cache()

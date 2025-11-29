# -*- coding: utf-8 -*-
"""AI Study Assistant - Corrected Version"""

# Step 1: Installing Required Libraries
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "langchain", "langchain-grpc", "langchain-community", "-y"])
subprocess.run([sys.executable, "-m", "pip", "install", 
                "langchain==0.1.20",
                "langchain-groq==0.1.4",
                "chromadb==1.3.5",
                "pypdf==4.2.0",
                "sentence-transformers==2.7.0",
                "langchain-community==0.0.38",
                "langchain-core==0.1.52",
                "gradio"])

# Step 2: Set API Keys
from google.colab import userdata
import os

api_key = userdata.get('APIKeytest')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key  # Fixed: was "TONKEN"
os.environ["GROQ_API_KEY"] = api_key

print("✓ Hugging Face Token Set!")
print("✓ Groq API Key Set!")

# Step 3: Import Libraries
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import gradio as gr

# Step 4: Define Functions
def load_pdf(pdf_path):
    """Load PDF file"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

def split_docs(docs):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(docs)

def create_db(chunks):
    """Create vector database using Chroma"""
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, emb)
    return db

def create_qa(db, llm):
    """Create RetrievalQA chain"""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

def summarize(text, llm):
    """Summarize text"""
    prompt = f"Summarize the following text in simple bullet points:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

def generate_mcqs(text, llm):
    """Generate multiple choice questions"""
    prompt = f"Generate 10 MCQs with answers from this text:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

# Step 5: Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

print("✓ AI Study Assistant Ready!")

# Step 6: Upload PDF
from google.colab import files

print("Uploading PDF...")
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

# Process PDF
print("Processing PDF...")
docs = load_pdf(pdf_path)
chunks = split_docs(docs)
db = create_db(chunks)
qa = create_qa(db, llm)

print("✓ PDF Loaded & Indexed!")

# Step 7: Create Gradio Interface
def qa_interface(question, history):
    """Q&A interface"""
    if question.strip():
        return qa.run(question)
    return "Please ask a question based on your notes."

def summarize_interface():
    """Summarizer interface"""
    text_to_summarize = " ".join([c.page_content for c in chunks[:5]])
    if text_to_summarize:
        return summarize(text_to_summarize, llm)
    return "PDF chunks not available for summarization."

def mcq_interface():
    """MCQ generator interface"""
    if chunks:
        return generate_mcqs(chunks[0].page_content, llm)
    return "PDF chunks not available for MCQ generation."

# Step 8: Build Gradio App
with gr.Blocks(title="AI Study Assistant") as demo:
    gr.Markdown("# 📚 AI Study Assistant Powered By GROQ")

    with gr.Tab("Ask Questions (RAG)"):
        gr.ChatInterface(
            fn=qa_interface,
            title="Question Answering from Notes",
            description="Ask specific questions about the content of your PDF.",
            submit_btn="Ask"
        )

    with gr.Tab("Summarizer"):
        gr.Markdown("### Summarize Key Content")
        summary_btn = gr.Button("Generate Summary of Notes")
        summary_output = gr.Textbox(label="Summary Output", lines=10)
        summary_btn.click(fn=summarize_interface, inputs=None, outputs=summary_output)

    with gr.Tab("MCQ Generator"):
        gr.Markdown("### Generate Practice MCQs")
        mcq_btn = gr.Button("Generate 10 MCQs")
        mcq_output = gr.Textbox(label="MCQ Output", lines=10)
        mcq_btn.click(fn=mcq_interface, inputs=None, outputs=mcq_output)

# Launch the app
demo.launch(share=True)

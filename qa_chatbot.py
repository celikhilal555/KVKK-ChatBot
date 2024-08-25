import streamlit as st
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Using a placeholder for embeddings
from langchain.chains import VectorDBQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import Field
import requests
import json

class OllamaQA(LLM):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model

    def generate_response(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt
        }
        print("detay",url)
        print(data)
        response = requests.post(url, json=data)
        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                data = json.loads(chunk)
                full_response += data["response"]
                if data["done"]:
                    break
        return full_response

    # Required by LangChain LLM interface
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.generate_response(prompt)

    @property
    def _identifying_params(self) -> dict:
        return {"base_url": self.base_url, "model": self.model}

    @property
    def _llm_type(self) -> str:
        return "ollama_qa"



def load_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def create_document(text):
    return Document(page_content=text)

def create_vector_store(documents, embeddings):
    # Create vector store with FAISS (in-memory vector store)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_qa_chain(vector_store, llm):
    qa_chain = VectorDBQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore=vector_store,
        return_source_documents=True
    )
    return qa_chain

def main():
    st.title("RAG Tabanli Soru-Cevap Chat Botu")
    #st.write("Upload a PDF file, and ask questions about the content!")
    st.write("Bir pdf dokumani yukleyin ve sorunuzu sorun!")

    # File uploader for PDF
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if pdf_file:
        text = load_pdf(pdf_file)
        document = create_document(text)
        
        # Initialize embeddings and FAISS vector store (using a placeholder for embeddings)
        embeddings = HuggingFaceEmbeddings()  # Use a Hugging Face model or other embeddings
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents([document])
        vector_store = create_vector_store(split_documents, embeddings)
        
        # Initialize your OllamaQA LLM
        ollama_qa = OllamaQA()
        
        # Create QA Chain using the custom LLM
        qa_chain = create_qa_chain(vector_store, ollama_qa)
        
        # Ask questions
        question = st.text_input("Dokuman Hakkindaki Sorunuzu Buraya Yaziniz:")
        if st.button("Soru Sor"):
            if question:
                response = ollama_qa.generate_response(question)
                st.write("**Cevap:**", response)
                # Optionally display the source documents
                result = qa_chain({"query": question})
                st.write("**Ilgili Dokuman Icerigi:**")
                for doc in result["source_documents"]:
                    st.write(doc.page_content[:500])  # Display the first 500 characters of the source document

if __name__ == "__main__":
    main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import base64
from PIL import Image
import io

load_dotenv(override=True)

# Initialize the multimodal model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt template for multimodal responses
prompt_template = """
Answer the following question based only on the provided context from PDFs and images.
Format your response as a poem - like the Percy Jackson prophecies - in French.

<context>
{context}
</context>

<image_descriptions>
{image_descriptions}
</image_descriptions>

<question>
{question}
</question>
"""

def encode_image(image_file):
    """Encode image to base64 for API calls"""
    if hasattr(image_file, 'read'):
        image_data = image_file.read()
    else:
        with open(image_file, "rb") as f:
            image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def extract_text_from_image(image_file):
    """Extract text from image using GPT-4V"""
    try:
        base64_image = encode_image(image_file)
        
        vision_prompt = """Please extract all text content from this image. 
        If it's a screenshot, extract any visible text. 
        If it's a photo of a document, extract the text as accurately as possible.
        Return only the extracted text."""
        
        response = llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ])
        return response.content
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return ""

def process_image_vision(image_file):
    """Process image and get description using GPT-4V"""
    try:
        base64_image = encode_image(image_file)
        
        vision_prompt = """Describe this image in detail for a RAG system.
        Include:
        - Any text visible in the image
        - Visual elements and their relationships
        - Charts, diagrams, or data visualizations
        - People, objects, or scenes
        Provide a comprehensive description that can be used for retrieval."""
        
        response = llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ])
        return response.content
    except Exception as e:
        st.error(f"Error processing image with vision: {e}")
        return ""

def main():
    st.set_page_config(page_title="Delphi Multimodal RAG Chat", layout="wide")
    st.subheader("Multimodal Retrieval Augmented Generation", divider="rainbow")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "image_context" not in st.session_state:
        st.session_state.image_context = []

    with st.sidebar:
        st.sidebar.title("📚 Document Upload")
        
        # Image upload
        img_path = r"C:\iaagentique\TP2\rag.png"
        if os.path.exists(img_path):
            st.image(img_path)
        else:
            st.info("✨ Upload your own logo/image above")
        
        # File uploaders
        st.markdown("### 📄 Upload Documents")
        pdf_docs = st.file_uploader(
            "Load PDFs", 
            type=['pdf'], 
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        image_files = st.file_uploader(
            "Load Images (JPG, PNG, etc.)", 
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True,
            key="image_uploader"
        )
        
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents and images..."):
                all_documents = []
                image_descriptions = []
                
                # Process PDFs
                if pdf_docs:
                    for pdf in pdf_docs:
                        content = ""
                        reader = PdfReader(pdf)
                        for page in reader.pages:
                            content += page.extract_text()
                        
                        # Split text
                        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            chunk_size=512, chunk_overlap=50
                        )
                        chunks = splitter.split_text(content)
                        
                        # Create documents with metadata
                        for i, chunk in enumerate(chunks):
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": pdf.name,
                                    "type": "pdf",
                                    "chunk": i
                                }
                            )
                            all_documents.append(doc)
                
                # Process Images
                if image_files:
                    progress_bar = st.progress(0)
                    for idx, img_file in enumerate(image_files):
                        # Extract text from image
                        extracted_text = extract_text_from_image(img_file)
                        if extracted_text:
                            doc = Document(
                                page_content=extracted_text,
                                metadata={
                                    "source": img_file.name,
                                    "type": "image",
                                    "extracted_text": True
                                }
                            )
                            all_documents.append(doc)
                        
                        # Get image description
                        description = process_image_vision(img_file)
                        if description:
                            desc_doc = Document(
                                page_content=description,
                                metadata={
                                    "source": img_file.name,
                                    "type": "image_description"
                                }
                            )
                            all_documents.append(desc_doc)
                            image_descriptions.append({
                                "file": img_file.name,
                                "description": description
                            })
                        
                        progress_bar.progress((idx + 1) / len(image_files))
                    
                    progress_bar.empty()
                
                # Create vector store
                if all_documents:
                    embedding_model = OpenAIEmbeddings()
                    vector_store = Chroma.from_documents(
                        all_documents,
                        embedding_model,
                        collection_name="multimodal_collection",
                    )
                    st.session_state.retriever = vector_store.as_retriever(
                        search_kwargs={"k": 8}
                    )
                    st.session_state.image_context = image_descriptions
                    st.success(f"✅ Processed {len(pdf_docs)} PDFs and {len(image_files)} images!")
                    st.info(f"Total chunks created: {len(all_documents)}")
                else:
                    st.warning("No documents were processed. Please upload files.")
    
    # Main chat interface
    st.markdown("## 🎯 Delphi Oracle Chatbot")
    st.caption("Ask questions about your PDFs and images - responses in French prophetic poems!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_question := st.chat_input("Ask your question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        if st.session_state.retriever:
            with st.chat_message("assistant"):
                with st.spinner("Consulting the Oracle..."):
                    # Retrieve relevant context
                    context_docs = st.session_state.retriever.invoke(user_question)
                    
                    # Separate PDF and image contexts
                    pdf_context = []
                    image_context = []
                    
                    for doc in context_docs:
                        if doc.metadata.get("type") == "image_description":
                            image_context.append(doc.page_content)
                        else:
                            pdf_context.append(doc.page_content)
                    
                    context_text = "\n\n".join(pdf_context[:5])
                    image_text = "\n\n".join(image_context[:3])
                    
                    # Create prompt
                    prompt = prompt_template.format(
                        context=context_text,
                        image_descriptions=image_text,
                        question=user_question
                    )
                    
                    # Get response
                    response = llm.invoke(prompt)
                    
                    # Display response
                    st.markdown(response.content)
                    
                    # Show sources in expander
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(context_docs[:5]):
                            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown(f"**Type:** {doc.metadata.get('type', 'text')}")
                            st.markdown(f"**Content preview:** {doc.page_content[:200]}...")
                            st.divider()
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
        else:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please upload and process documents in the sidebar first!")

if __name__ == "__main__":
    main()
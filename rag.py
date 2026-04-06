import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt_template = """
Answer the following question based only on the provided context in the form of a poem - like the Percy Jackson prophecies - in french

<context>
{context}
</context>
<question>
{question}
</question>
"""

def main():
    st.set_page_config(page_title="Delphi reg chat", layout="wide")
    st.subheader("Retrieval Augmented Generation", divider="blue")

    with st.sidebar:
        st.sidebar.title("chat nul PT")
        
        # Vérification de l'image
        img_path = r"C:\iaagentique\TP2\rag.png"
        if os.path.exists(img_path):
            st.image(img_path)
        else:
            st.warning("Image introuvable : vérifie le chemin ou le nom du fichier.")

        pdf_docs = st.file_uploader(label="Load your PDFs", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Loading"):
                content = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        content += page.extract_text()

                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=512, chunk_overlap=16
                )
                chunks = splitter.split_text(content)
                st.write(chunks)

                embedding_model = OpenAIEmbeddings()
                vector_store = Chroma.from_texts(
                    chunks,
                    embedding_model,
                    collection_name="data_collection",
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                st.session_state.retriever = retriever

    st.subheader("Delphi chatbot")
    user_question = st.text_input("Ask your question")
    if user_question and "retriever" in st.session_state:
        context_docs = st.session_state.retriever.invoke(user_question)
        context_list = [d.page_content for d in context_docs]
        context_text = ".".join(context_list)

        prompt = prompt_template.format(context=context_text, question=user_question)
        resp = llm.invoke(prompt)

        st.write(resp.content)

if __name__ == "__main__":
    main()

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

st.set_page_config(page_title="Assistente de Doutrina Jurídica", page_icon="📚")
st.title("📘 Assistente de Pesquisa Jurídica")
st.markdown("Consulte a doutrina carregada e obtenha respostas com base nos textos PDF do repositório.")

openai_api_key = st.text_input("🔐 Sua OpenAI API Key", type="password")

@st.cache_resource(show_spinner=True)
def carregar_qa_chain(api_key):
    # Carregar e processar todos os PDFs da pasta 'doutrina'
    folder_path = "doutrina"
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    documents = []
    for file in pdf_files:
        loader = PyMuPDFLoader(os.path.join(folder_path, file))
        docs = loader.load()
        documents.extend(docs)

    # Fragmentar os textos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_documents(documents)

    # Criar vetor e modelo
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = Chroma.from_documents(texts, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6})
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", openai_api_key=api_key),
        chain_type="refine",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

if openai_api_key:
    qa_chain = carregar_qa_chain(openai_api_key)
    pergunta = st.text_input("✏️ Digite sua pergunta doutrinária:")
    if st.button("🔍 Consultar") and pergunta:
        with st.spinner("Buscando na doutrina..."):
            resposta = qa_chain(pergunta)
            st.markdown("### 📚 Resposta")
            st.write(resposta["result"])
            st.markdown("---")
            st.markdown("### 📎 Fontes consultadas")
            for doc in resposta["source_documents"]:
                st.markdown(f"- `{doc.metadata.get('source', 'sem nome')}`")
else:
    st.warning("Por favor, insira sua OpenAI API Key para usar o assistente.")

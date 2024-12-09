import streamlit as st
from config_reader import config
from src.ColQwen2ForRAGLangchain import ColQwen2ForRAGLangchain
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pathlib import Path
from src.pdf2image_loader import Pdf2ImageLoader

@st.cache_resource
def init_openai_client():
    return OpenAI(
        base_url=config.LLM_BASE_URL.get_secret_value(),
        api_key=config.LLM_API_KEY.get_secret_value(),
    )

# Initialize embeddings
@st.cache_resource
def init_colqwen2():
    return ColQwen2ForRAGLangchain()

# Initialize all resources
client = init_openai_client()
embeddings = init_colqwen2().ImageEmbeddings

# Streamlit UI
st.title("Норникель PDF Ассистент")

# Sidebar with file operations
with st.sidebar:
    st.header("Управление документами")
    
    # Upload PDF
    uploaded_file = st.file_uploader("Загрузить PDF документ", type=['pdf'])
    if uploaded_file:
        if uploaded_file.name in st.session_state.get("DB_LIST", []):
            st.warning("Этот файл уже был загружен")
        else:
            with st.spinner("Обработка документа..."):
                try:
                    # Save temporary file
                    temp_file_path = uploaded_file.name
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    
                    # Load and process the PDF
                    loader = Pdf2ImageLoader(temp_file_path)
                    documents = loader.load()
                    
                    # Add documents to vector store
                    vector_store = QdrantVectorStore.from_documents(documents, embeddings, 
                                                                    collection_name=config.QDRANT_COLLECTION_NAME.get_secret_value(),
                                                                    url=config.QDRANT_URL.get_secret_value(),
                                                                    api_key=config.QDRANT_API_KEY.get_secret_value(),
                                                                    validate_collection_config=False)
                    
                    st.session_state["DB_LIST"] = st.session_state.get("DB_LIST", []).append(uploaded_file.name)
                    st.success(f"Файл {uploaded_file.name} успешно загружен и проиндексирован!")
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке файла: {str(e)}")
                finally:
                    if Path(temp_file_path).exists():
                        Path(temp_file_path).unlink()

    # List indexed documents
    st.header("Проиндексированные документы")
    if st.session_state.get("DB_LIST", []):
        for filename in sorted(st.session_state.get("DB_LIST", [])):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(filename)
    else:
        st.write("Нет загруженных документов")

# Main chat interface
st.header("Чат с ассистентом")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "images" in message:
            for img in message["images"]:
                st.image(img)
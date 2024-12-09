import base64
from io import BytesIO
from PIL import Image
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

vector_store = None

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
                                                                    api_key=config.QDRANT_API_KEY.get_secret_value())
                    
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

# Chat input
if prompt := st.chat_input("Задайте вопрос..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            # Get context from retrievers
            if vector_store is None:
                vector_store = QdrantVectorStore.from_existing_collection(collection_name=config.QDRANT_COLLECTION_NAME.get_secret_value(),
                                                                embedding=embeddings,
                                                                url=config.QDRANT_URL.get_secret_value(),
                                                                api_key=config.QDRANT_API_KEY.get_secret_value())
            results_image = vector_store.similarity_search_by_vector(init_colqwen2().TextEmbeddings.embed_query(prompt), k=5)
            
            # Process image results
            images = []
            for result in results_image:
                try:
                    image_base64 = result.page_content
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',')[1]
                    padding = 4 - (len(image_base64) % 4) if len(image_base64) % 4 else 0
                    image_base64 = image_base64 + ('=' * padding)
                    
                    img_data = base64.b64decode(image_base64)
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                except Exception as e:
                    st.error(f"Ошибка обработки изображения: {str(e)}")

            if images:
                st.write("Найденные изображения:")
                cols = st.columns(min(len(images), 3))
                for idx, (img, col) in enumerate(zip(images, cols)):
                    col.image(img, caption=f"Изображение {idx + 1}")

import streamlit as st
import logging
import PIL
from config_reader import config
from src.ColQwen2Embeddings import ColQwen2Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models, QdrantClient
from langchain.retrievers.ensemble import EnsembleRetriever
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import zlib
from pathlib import Path
from src.NornikelPdfLoader import NornikelPdfLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants
CHUNK_SIZE = 1 * 1024 * 1024      # 1MB chunks

# Initialize session state
if 'DB_LIST' not in st.session_state:
    st.session_state.DB_LIST = []  # Changed from dict to list

# Qdrant client initialization
@st.cache_resource
def init_qdrant_client():
    return QdrantClient(
        url=config.QDRANT_URL.get_secret_value(),
        api_key=config.QDRANT_API_KEY.get_secret_value(),
    )

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    return OpenAI(
        base_url=config.LLM_BASE_URL.get_secret_value(),
        api_key=config.LLM_API_KEY.get_secret_value(),
    )

# Initialize embeddings
@st.cache_resource
def init_embeddings():
    return ColQwen2Embeddings()

# Initialize Qdrant vector store
@st.cache_resource
def init_vector_store(_embeddings):  # Добавлено подчеркивание
    qdrant = QdrantVectorStore.from_documents(
        documents=[],
        embedding=_embeddings,  # Используем параметр с подчеркиванием
        url=config.QDRANT_URL.get_secret_value(),
        api_key=config.QDRANT_API_KEY.get_secret_value(),
        collection_name=config.QDRANT_COLLECTION_NAME.get_secret_value(),
        force_recreate=True
    )
    return qdrant

# Initialize retrievers
@st.cache_resource
def init_retrievers(_qdrant):  # Добавлено подчеркивание
    retriever_text = _qdrant.as_retriever(  # Используем параметр с подчеркиванием
        search_type="mmr",
        search_kwargs={
            "k": 7,
            "filter": models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchValue(value="text")
                    )
                ]
            )
        }
    )

    retriever_image = _qdrant.as_retriever(  # Используем параметр с подчеркиванием
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "filter": models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchValue(value="image")
                    )
                ]
            )
        }
    )
    
    return retriever_text, retriever_image

# Initialize all resources
qdrant_client = init_qdrant_client()
client = init_openai_client()
embeddings = init_embeddings()
qdrant = init_vector_store(embeddings)
retriever_text, retriever_image = init_retrievers(qdrant)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Streamlit UI
st.title("Норникель PDF Ассистент")

# Sidebar with file operations
with st.sidebar:
    st.header("Управление документами")
    
    # Upload PDF
    uploaded_file = st.file_uploader("Загрузить PDF документ", type=['pdf'])
    if uploaded_file:
        if uploaded_file.name in st.session_state.DB_LIST:
            st.warning("Этот файл уже был загружен")
        else:
            with st.spinner("Обработка документа..."):
                try:
                    # Save temporary file
                    temp_file_path = uploaded_file.name
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    
                    # Load and process the PDF
                    loader = NornikelPdfLoader(temp_file_path)
                    documents = loader.load()
                    
                    # Split documents into chunks
                    chunks = text_splitter.split_documents(documents)
                    
                    # Add documents to vector store
                    qdrant.add_documents(chunks)
                    
                    st.session_state.DB_LIST.append(uploaded_file.name)
                    st.success(f"Файл {uploaded_file.name} успешно загружен и проиндексирован!")
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке файла: {str(e)}")
                finally:
                    if Path(temp_file_path).exists():
                        Path(temp_file_path).unlink()

    # List indexed documents
    st.header("Проиндексированные документы")
    if st.session_state.DB_LIST:
        for filename in sorted(st.session_state.DB_LIST):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(filename)
            with col2:
                if st.button("Удалить", key=filename):
                    try:
                        # Delete from Qdrant
                        qdrant_client.delete(
                            collection_name=config.QDRANT_COLLECTION_NAME.get_secret_value(),
                            points_selector=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="metadata.source",
                                        match=models.MatchValue(value=filename)
                                    )
                                ]
                            )
                        )
                        # Remove from session state
                        st.session_state.DB_LIST.remove(filename)
                        st.success(f"Документ {filename} удален")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка при удалении: {str(e)}")
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
            results_text = retriever_text.invoke(prompt)
            results_image = retriever_image.invoke(prompt)

            # Process text results
            texts = [f"{result.metadata['source']}/{result.metadata['page']} стр./{result.page_content}" 
                    for result in results_text if result.metadata['type'] == 'text']
            
            # Process image results
            images = []
            for result in results_image:
                if result.metadata['type'] == 'image':
                    try:
                        image_base64 = result.metadata['image_base64']
                        if ',' in image_base64:
                            image_base64 = image_base64.split(',')[1]
                        padding = 4 - (len(image_base64) % 4) if len(image_base64) % 4 else 0
                        image_base64 = image_base64 + ('=' * padding)
                        
                        img_data = base64.b64decode(image_base64)
                        img = Image.open(BytesIO(img_data))
                        images.append(img)
                    except Exception as e:
                        st.error(f"Ошибка обработки изображения: {str(e)}")

            # Display context
            if texts:
                st.write("Найденный контекст:")
                for idx, text in enumerate(texts, 1):
                    st.info(f"Контекст {idx}: {text}")

            if images:
                st.write("Найденные изображения:")
                cols = st.columns(min(len(images), 3))
                for idx, (img, col) in enumerate(zip(images, cols)):
                    col.image(img, caption=f"Изображение {idx + 1}")

            # Get LLM response
            rag_context = "\n\n".join(texts)
            completion = client.chat.completions.create(
                model="Qwen/Qwen2-VL-72B-Instruct-AWQ",
                messages=[
                    {"role": "system", "content": """Я - виртуальный ассистент компании «Норникель»..."""},
                    {"role": "system", "content": rag_context},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
            )
            
            response = completion.choices[0].message.content
            st.write(response)

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "images": images if images else []
            })

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

# Initialize session state for retriever parameters
if 'retriever_text_k' not in st.session_state:
    st.session_state.retriever_text_k = 7
if 'retriever_image_k' not in st.session_state:
    st.session_state.retriever_image_k = 3

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
def init_retrievers(_qdrant):  # Добавлено подчеркивание
    retriever_text = _qdrant.as_retriever(  # Используем параметр с подчеркиванием
        search_type="mmr",
        search_kwargs={
            "k": st.session_state.retriever_text_k,
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
            "k": st.session_state.retriever_image_k,
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
client = init_openai_client()
embeddings = init_embeddings()
qdrant = init_vector_store(embeddings)
retriever_text, retriever_image = init_retrievers(qdrant)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Qdrant client initialization
@st.cache_resource
def init_qdrant_client():
    return QdrantClient(
        url=config.QDRANT_URL.get_secret_value(),
        api_key=config.QDRANT_API_KEY.get_secret_value(),
    )

qdrant_client = init_qdrant_client()

# Initialize session state
if 'DB_LIST' not in st.session_state:
    # Получаем все точки и собираем уникальные source
    unique_sources = set()
    offset = None

    while True:
        search_result = qdrant_client.scroll(
            collection_name='nornikel_2024_team_hakuna_matata',
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )
        
        points, offset = search_result  # распаковываем результат
        
        # Если получили пустой список точек, выходим из цикла
        if not points:
            break
        
        # Добавляем source в множество
        for point in points:
            if 'metadata' in point.payload and 'source' in point.payload['metadata']:
                unique_sources.add(point.payload['metadata']['source'])
        
        # Если offset None, значит это была последняя страница
        if offset is None:
            break
    st.session_state.DB_LIST = list(unique_sources)  # Changed from dict to list

# Streamlit UI
st.title("Норникель PDF Ассистент")

# Sidebar with file operations
with st.sidebar:
    # Add retriever parameters inputs
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.retriever_text_k = st.number_input(
            "top-k текст",
            min_value=1,
            max_value=20,
            value=st.session_state.retriever_text_k
        )
    with col2:
        st.session_state.retriever_image_k = st.number_input(
            "top-k фото",
            min_value=1,
            max_value=10,
            value=st.session_state.retriever_image_k
        )

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
            
            # Prepare messages for the API
            messages = [
                {"role": "system", "content": """Я - виртуальный ассистент компании «Норникель», созданный для помощи с документами и информацией компании. Я использую мультимодальную RAG-систему для поиска и анализа релевантной информации в корпоративных документах, изображениях и других материалах.

                Я отвечаю на вопросы, основываясь на предоставленном контексте из базы знаний.

                Мои ответы всегда вежливы, профессиональны и соответствуют корпоративной культуре Норникеля."""},
                {"role": "system", "content": rag_context},
                {"role": "user", "content": prompt}
            ]

            # Only add image if available
            if images:
                # Convert PIL Image to base64
                buffered = BytesIO()
                # Convert RGBA to RGB if necessary
                if images[0].mode == 'RGBA':
                    images[0] = images[0].convert('RGB')
                images[0].save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                messages[-1]["content"] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]

            completion = client.chat.completions.create(
                model="Qwen/Qwen2-VL-72B-Instruct-AWQ",
                messages=messages
            )
            
            response = completion.choices[0].message.content
            st.write(response)

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "images": images if images else []
            })

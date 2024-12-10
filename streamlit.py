import streamlit as st
import httpx
import asyncio
from functools import wraps
from PIL import Image

# API configuration
API_BASE_URL = "http://localhost:8000"  # Adjust as needed

# Helper to run async functions in Streamlit
def async_to_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

async def upload_file(file) -> dict:
    async with httpx.AsyncClient() as client:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = await client.post(f"{API_BASE_URL}/documents/upload", files=files)
        response.raise_for_status()
        return response.json()

async def get_documents() -> list:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/documents")
        response.raise_for_status()
        return response.json()

async def delete_document(filename: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{API_BASE_URL}/documents/{filename}")
        response.raise_for_status()
        return response.json()

async def chat_request(message: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/chat",
            json={"message": message}
        )
        response.raise_for_status()
        return response.json()
    
def base64_to_image(text: str) -> Image.Image:
    try:
        import base64
        from io import BytesIO
        
        image_base64 = text
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        padding = 4 - (len(image_base64) % 4) if len(image_base64) % 4 else 0
        image_base64 = image_base64 + ('=' * padding)
        img_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(img_data))
    except Exception:
        raise ValueError(f"Ошибка обработки изображения. Введите изображение в формате base64. Ваш ввод: {text}")

# Streamlit UI
st.title("Норникель PDF Ассистент")

# Sidebar with file operations
with st.sidebar:
    st.header("Управление документами")
    
    # Upload PDF
    uploaded_file = st.file_uploader("Загрузить PDF документ", type=['pdf'])
    if uploaded_file:
        with st.spinner("Обработка документа..."):
            try:
                response = async_to_sync(upload_file)(uploaded_file)
                st.success(response["message"])
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")

    # List indexed documents
    st.header("Проиндексированные документы")
    try:
        documents = async_to_sync(get_documents)()
        
        if documents:
            for filename in sorted(documents):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(filename)
                with col2:
                    if st.button("Удалить", key=filename):
                        try:
                            response = async_to_sync(delete_document)(filename)
                            st.success(response["message"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка при удалении: {str(e)}")
        else:
            st.write("Нет загруженных документов")
    except Exception as e:
        st.error(f"Ошибка при получении списка документов: {str(e)}")

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

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                chat_response = async_to_sync(chat_request)(prompt)
                
                # Display images if present
                if "context_images" in chat_response:
                    for index, img_data in enumerate(chat_response["context_images"]):
                        image = base64_to_image(img_data)
                        st.image(image, caption=f"{chat_response['sources'][index]} / {chat_response['pages'][index]} стр.")
                
                # Display text response
                st.write(chat_response["response"])
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": chat_response["response"],
                    "images": chat_response.get("images", [])
                })
            except Exception as e:
                st.error(f"Ошибка при получении ответа: {str(e)}")
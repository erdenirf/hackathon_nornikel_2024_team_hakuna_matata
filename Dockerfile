# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies including CUDA related packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.3
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
uvicorn main:app --host 0.0.0.0 --port 8000 & \n\
streamlit run streamlit.py --server.port 8501 --server.address 0.0.0.0\n\
wait' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
# Use a base image with Python and common packages
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy your local files into the container
COPY scripts/ ./scripts/
COPY dino3d/utils/colmap_utils.py ./dino3d/utils/colmap_utils.py

# Set the PYTHONPATH so Python can find dino3d
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    matplotlib \
    tqdm \
    numpy \
    Pillow

# Set the default command to bash so you can interact
CMD ["/bin/bash"]


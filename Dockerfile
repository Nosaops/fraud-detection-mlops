# Start from an official Python image (slim = smaller size)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (Docker caches this layer)
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your model file into the container
COPY gs_rf.pkl .

# Copy your API code into the container
COPY fraudapp.py .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "fraudapp:app", "--host", "0.0.0.0", "--port", "8000"]

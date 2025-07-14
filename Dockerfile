# base image
FROM python:3.12

# system dependencies
RUN apt-get update && apt-get install -y \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# working directory
WORKDIR /app

# Copy 
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Unzip FAISS index
RUN unzip faiss_index_interview_questions.zip -d faiss_index_interview_questions

# Expose
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "dogshit.py", "--server.port=8501", "--server.address=0.0.0.0"]

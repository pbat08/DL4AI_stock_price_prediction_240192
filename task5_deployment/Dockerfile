FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project (models, API script)
COPY . .

EXPOSE 5000
CMD ["python", "[task5.1]_api_server_.py"]

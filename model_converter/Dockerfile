FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Create data directory
RUN mkdir -p /app/data

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Create and set permission for the entrypoint script
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Define a volume for persistent storage
VOLUME /app/data

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
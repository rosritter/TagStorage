FROM chromadb/chroma:0.6.3

# Set only essential environment variables
ENV CHROMA_SERVER_HOST=0.0.0.0
ENV CHROMA_SERVER_HTTP_PORT=8000

# Directory for persistent storage
VOLUME /chroma/chroma

# Expose the port
EXPOSE 8000
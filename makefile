.PHONY: setup build up down clean

# Create necessary directories and copy .env files
setup:
	mkdir -p fastapi/models
	cp .env model_converter/
	cp .env fastapi/
	cp .env chromadb/

# Build images
build:
	docker compose build

# Start services
up:
	docker compose up -d

# Stop services
down:
	docker compose down

# Clean everything
clean: down
	docker rmi embeddings-model-converter embeddings-chroma embeddings-fastapi
	rm -f model_converter/.env fastapi/.env chromadb/.env

# Show logs
logs:
	docker compose logs -f
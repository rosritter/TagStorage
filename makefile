.PHONY: setup build up down clean

# Copy .env files to directories
setup:
	cp .env model_converter/
	cp .env fastapi/
	cp .env chromadb/

# Build images
build:
	docker-compose build

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# Clean everything
clean: down
	docker rmi embeddings-model-converter embeddings-chroma embeddings-fastapi
# rm model_converter/.env fastapi/.env chromadb/.env

# Show logs
logs:
	docker-compose logs -f
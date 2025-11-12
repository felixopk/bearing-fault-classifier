.PHONY: help build build-slim build-prod run stop logs test clean

help:
	@echo "Available commands:"
	@echo "  make build-slim    - Build slim image (with shell, for dev)"
	@echo "  make build-prod    - Build distroless image (production)"
	@echo "  make run           - Run container"
	@echo "  make stop          - Stop container"
	@echo "  make logs          - View logs"
	@echo "  make test          - Test endpoints"
	@echo "  make clean         - Clean up"

build-slim:
	docker build -f Dockerfile.slim -t bearing-fault-classifier:slim .

build-prod:
	docker build -t bearing-fault-classifier:distroless .

build: build-prod

run:
	docker-compose up -d

stop:
	docker-compose down

logs:
	docker-compose logs -f

test:
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8000/health
	@echo "\nTesting model info..."
	@curl -f http://localhost:8000/model/info

clean:
	docker-compose down -v
	docker rmi bearing-fault-classifier:distroless || true
	docker rmi bearing-fault-classifier:slim || true

compare-sizes:
	@echo "Image sizes:"
	@docker images | grep bearing-fault-classifier

shell-slim:
	docker run -it --rm bearing-fault-classifier:slim /bin/bash

# Can't shell into distroless (no shell!)
inspect-prod:
	docker run -it --rm --entrypoint sh gcr.io/distroless/python3-debian11:debug

#!/bin/bash

# Fraud Detection Docker Container Management Script


set -e  # Exit on any error

# Configuration
IMAGE_NAME="fraud-detection-model"
IMAGE_TAG="v1"
CONTAINER_NAME="fraud-detection-api"
PORT="5000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running!"
        exit 1
    fi

    print_success "Docker is ready"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

    # Record build start time
    BUILD_START=$(date +%s)

    # Build image with progress
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} . --progress=plain

    # Calculate build time
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))

    print_success "Image built successfully in ${BUILD_TIME} seconds"

    # Show image info
    IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${IMAGE_TAG} --format "table {{.Size}}" | tail -n 1)
    print_status "Image size: ${IMAGE_SIZE}"
}

# Function to run container
run_container() {
    print_status "Running container: ${CONTAINER_NAME}"

    # Stop existing container if running
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_warning "Stopping existing container..."
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
    fi

    # Run new container
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:5000 \
        --restart unless-stopped \
        ${IMAGE_NAME}:${IMAGE_TAG}

    print_success "Container started successfully"
    print_status "Container available at: http://localhost:${PORT}"
}

# Function to check container health
check_health() {
    print_status "Checking container health..."

    # Wait for container to be ready
    MAX_WAIT=60
    COUNT=0

    while [ $COUNT -lt $MAX_WAIT ]; do
        if curl -f http://localhost:${PORT}/health &> /dev/null; then
            print_success "Container is healthy and ready!"
            return 0
        fi

        echo -n "."
        sleep 1
        COUNT=$((COUNT + 1))
    done

    print_error "Container failed to become healthy within ${MAX_WAIT} seconds"
    return 1
}

# Function to show container logs
show_logs() {
    print_status "Showing container logs..."
    docker logs ${CONTAINER_NAME} --tail 50 -f
}

# Function to stop container
stop_container() {
    print_status "Stopping container: ${CONTAINER_NAME}"

    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
        print_success "Container stopped and removed"
    else
        print_warning "Container is not running"
    fi
}

# Function to show container stats
show_stats() {
    print_status "Container resource usage:"
    docker stats ${CONTAINER_NAME} --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Function to run tests
run_tests() {
    print_status "Running API tests..."

    if [ -f "test_docker.py" ]; then
        python test_docker.py
    else
        print_error "test_docker.py not found!"
        exit 1
    fi
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."

    # Stop and remove container
    stop_container

    # Remove image
    if docker images -q ${IMAGE_NAME}:${IMAGE_TAG} | grep -q .; then
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG}
        print_success "Image removed"
    fi

    # Clean up unused resources
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "Fraud Detection Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  run        Run the container"
    echo "  stop       Stop and remove the container"
    echo "  logs       Show container logs"
    echo "  stats      Show container resource usage"
    echo "  health     Check container health"
    echo "  test       Run API tests"
    echo "  restart    Stop and restart the container"
    echo "  cleanup    Clean up all Docker resources"
    echo "  full       Full build, run, and test cycle"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build      # Build the image"
    echo "  $0 full       # Complete deployment"
    echo "  $0 test       # Run tests"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        run_container
        check_health
        ;;
    "stop")
        check_docker
        stop_container
        ;;
    "logs")
        check_docker
        show_logs
        ;;
    "stats")
        check_docker
        show_stats
        ;;
    "health")
        check_docker
        check_health
        ;;
    "test")
        run_tests
        ;;
    "restart")
        check_docker
        stop_container
        run_container
        check_health
        ;;
    "cleanup")
        check_docker
        cleanup
        ;;
    "full")
        check_docker
        build_image
        run_container
        check_health
        run_tests
        ;;
    "help"|*)
        show_help
        ;;
esac

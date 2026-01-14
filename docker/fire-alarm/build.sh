#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'USAGE'
用法: docker/fire-alarm/build.sh [选项]

选项:
  -f, --file DOCKERFILE   指定 Dockerfile（默认 docker/fire-alarm/Dockerfile）
  -t, --tag TAG           镜像名（默认 fire-alarm-consumer）
  -p, --platform PLAT     目标架构（默认 linux/amd64）
  -h, --help              显示帮助

示例:
  docker/fire-alarm/build.sh -f docker/fire-alarm/Dockerfile.cuda -t fire-alarm-consumer:cuda -p linux/amd64
  docker/fire-alarm/build.sh -f docker/fire-alarm/Dockerfile -t fire-alarm-consumer -p linux/arm64
USAGE
}

DOCKERFILE="docker/fire-alarm/Dockerfile"
TAG="fire-alarm-consumer"
PLATFORM="linux/amd64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      DOCKERFILE="$2"
      shift 2
      ;;
    -t|--tag)
      TAG="$2"
      shift 2
      ;;
    -p|--platform)
      PLATFORM="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      show_help
      exit 1
      ;;
  esac
 done

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "未找到 Dockerfile: $DOCKERFILE" >&2
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "未检测到 buildx，请先安装/启用 Docker buildx" >&2
  exit 1
fi

if [[ "$PLATFORM" == *","* ]]; then
  echo "仅支持单一架构（--platform linux/amd64 或 linux/arm64）" >&2
  exit 1
fi

DOCKER_BUILDKIT=1 docker buildx build \
  -f "$DOCKERFILE" \
  -t "$TAG" \
  --platform "$PLATFORM" \
  --load \
  .

#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'USAGE'
用法: kafuka/labcoat/docker/build.sh [选项]

选项:
  -f, --file DOCKERFILE   指定 Dockerfile（默认 kafuka/labcoat/docker/Dockerfile）
  -t, --tag TAG           镜像完整标签（默认 labcoat-consumer）
  -n, --name NAME         镜像名称（与 --tag-name 组合）
  --tag-name TAG          镜像标签（与 --name 组合）
  -p, --platform PLAT     目标架构（默认 linux/amd64）
  -c, --cache-dir DIR     Buildx 本地缓存目录（默认 .docker-cache/labcoat-consumer）
  -h, --help              显示帮助

示例:
  kafuka/labcoat/docker/build.sh -f kafuka/labcoat/docker/Dockerfile.cuda -t labcoat-consumer:cuda -p linux/amd64
  kafuka/labcoat/docker/build.sh -f kafuka/labcoat/docker/Dockerfile -n labcoat-consumer --tag-name v1 -p linux/arm64
USAGE
}

DOCKERFILE="kafuka/labcoat/docker/Dockerfile"
TAG="labcoat-consumer"
NAME=""
TAG_NAME=""
PLATFORM="linux/amd64"
CACHE_DIR=".docker-cache/labcoat-consumer"

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
    -n|--name)
      NAME="$2"
      shift 2
      ;;
    --tag-name)
      TAG_NAME="$2"
      shift 2
      ;;
    -p|--platform)
      PLATFORM="$2"
      shift 2
      ;;
    -c|--cache-dir)
      CACHE_DIR="$2"
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

if [[ -n "$NAME" ]]; then
  if [[ -z "$TAG_NAME" ]]; then
    TAG_NAME="latest"
  fi
  TAG="${NAME}:${TAG_NAME}"
fi

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
  --cache-from "type=local,src=$CACHE_DIR" \
  --cache-to "type=local,dest=$CACHE_DIR,mode=max" \
  --load \
  .

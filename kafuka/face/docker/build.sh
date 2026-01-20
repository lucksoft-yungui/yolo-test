#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'USAGE'
用法: docker/build.sh [选项]

选项:
  -f, --file DOCKERFILE   指定 Dockerfile（默认 docker/Dockerfile）
  -t, --tag TAG           镜像完整标签（默认 face-consumer）
  -n, --name NAME         镜像名称（与 --tag-name 组合）
  --tag-name TAG          镜像标签（与 --name 组合）
  -p, --platform PLAT     目标架构（默认 linux/amd64）
  -c, --cache-dir DIR     Buildx 本地缓存目录（默认 .docker-cache/face-consumer）
  -C, --context DIR       构建上下文目录（默认 .）
  -a, --build-arg ARG     透传 build-arg（可重复）
  --add-host HOST         透传 add-host（可重复）
  -h, --help              显示帮助

示例:
  docker/build.sh -f docker/Dockerfile.cuda -t face-consumer:cuda -p linux/amd64
  docker/build.sh -f docker/Dockerfile -n face-consumer --tag-name v1 -p linux/arm64
  docker/build.sh -f docker/Dockerfile.cuda -n face-consumer --tag-name v1 -p linux/amd64 \
    -a http_proxy=http://host.docker.internal:7890 \
    -a https_proxy=http://host.docker.internal:7890 \
    -a all_proxy=socks5://host.docker.internal:7890 \
    --add-host host.docker.internal:host-gateway
USAGE
}

DOCKERFILE="docker/Dockerfile"
TAG="face-consumer"
NAME=""
TAG_NAME=""
PLATFORM="linux/amd64"
CACHE_DIR=".docker-cache/face-consumer"
CONTEXT_DIR="."
BUILD_ARGS=()
ADD_HOSTS=()

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
    -C|--context)
      CONTEXT_DIR="$2"
      shift 2
      ;;
    -a|--build-arg)
      BUILD_ARGS+=("$2")
      shift 2
      ;;
    --add-host)
      ADD_HOSTS+=("$2")
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

BUILD_ARG_FLAGS=()
if (( ${#BUILD_ARGS[@]} )); then
  for arg in "${BUILD_ARGS[@]}"; do
    BUILD_ARG_FLAGS+=(--build-arg "$arg")
  done
fi

ADD_HOST_FLAGS=()
if (( ${#ADD_HOSTS[@]} )); then
  for host in "${ADD_HOSTS[@]}"; do
    ADD_HOST_FLAGS+=(--add-host "$host")
  done
fi

export DOCKER_BUILDKIT=1

CMD=(docker buildx build
  -f "$DOCKERFILE"
  -t "$TAG"
  --platform "$PLATFORM"
  --cache-from "type=local,src=$CACHE_DIR"
  --cache-to "type=local,dest=$CACHE_DIR,mode=max"
)

if (( ${#BUILD_ARG_FLAGS[@]} )); then
  CMD+=("${BUILD_ARG_FLAGS[@]}")
fi

if (( ${#ADD_HOST_FLAGS[@]} )); then
  CMD+=("${ADD_HOST_FLAGS[@]}")
fi

CMD+=(--load "$CONTEXT_DIR")

"${CMD[@]}"

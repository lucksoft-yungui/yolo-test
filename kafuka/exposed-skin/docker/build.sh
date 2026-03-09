#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'USAGE'
用法: kafuka/exposed-skin/docker/build.sh [选项]

选项:
  -f, --file DOCKERFILE   指定 Dockerfile（默认 kafuka/exposed-skin/docker/Dockerfile）
  -t, --tag TAG           镜像完整标签（默认 exposed-skin-consumer）
  -n, --name NAME         镜像名称（与 --tag-name 组合）
  --tag-name TAG          镜像标签（与 --name 组合）
  -p, --platform PLAT     目标架构（默认 linux/amd64）
  -c, --cache-dir DIR     Buildx 本地缓存目录（默认 .docker-cache/exposed-skin-consumer）
  --base-image IMAGE      复用基础镜像（传递 BASE_IMAGE 构建参数）
  -b, --builder NAME      指定 buildx builder（默认 default）
  -h, --help              显示帮助

示例:
  kafuka/exposed-skin/docker/build.sh -f kafuka/exposed-skin/docker/Dockerfile.cuda -t exposed-skin-consumer:cuda -p linux/amd64
  kafuka/exposed-skin/docker/build.sh -f kafuka/exposed-skin/docker/Dockerfile -n exposed-skin-consumer --tag-name v1 -p linux/arm64
USAGE
}

DOCKERFILE="kafuka/exposed-skin/docker/Dockerfile"
TAG="exposed-skin-consumer"
NAME=""
TAG_NAME=""
PLATFORM="linux/amd64"
CACHE_DIR=".docker-cache/exposed-skin-consumer"
BASE_IMAGE=""
BUILDER="default"

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
    --base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    -b|--builder)
      BUILDER="$2"
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

BUILD_ARGS=()
for key in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
  if [[ -n "${!key:-}" ]]; then
    BUILD_ARGS+=(--build-arg "$key=${!key}")
  fi
done
if [[ -n "$BASE_IMAGE" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
fi

INSPECT_OUTPUT="$(docker buildx inspect "$BUILDER" 2>/dev/null || true)"
DRIVER="$(printf '%s\n' "$INSPECT_OUTPUT" | sed -n 's/^Driver:[[:space:]]*//p' | head -n1)"
BUILDER_ENDPOINT="$(printf '%s\n' "$INSPECT_OUTPUT" | sed -n 's/^Endpoint:[[:space:]]*//p' | head -n1)"
if [[ -z "$DRIVER" ]]; then
  echo "未找到 builder: $BUILDER" >&2
  exit 1
fi

CACHE_FLAGS=()
if [[ "$DRIVER" == "docker-container" ]]; then
  CACHE_FLAGS+=(--cache-from "type=local,src=$CACHE_DIR")
  CACHE_FLAGS+=(--cache-to "type=local,dest=$CACHE_DIR,mode=max")
fi

DOCKER_CMD=(docker)
if [[ -n "$BUILDER_ENDPOINT" ]]; then
  DOCKER_CMD+=(--context "$BUILDER_ENDPOINT")
fi

CMD=("${DOCKER_CMD[@]}" buildx build
  --builder "$BUILDER"
  -f "$DOCKERFILE"
  -t "$TAG"
  --platform "$PLATFORM"
  --load
)

if ((${#CACHE_FLAGS[@]})); then
  CMD+=("${CACHE_FLAGS[@]}")
fi
if ((${#BUILD_ARGS[@]})); then
  CMD+=("${BUILD_ARGS[@]}")
fi
CMD+=(.)

DOCKER_BUILDKIT=1 "${CMD[@]}"

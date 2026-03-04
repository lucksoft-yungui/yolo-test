FROM python:3.11

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl python3-dev build-essential gfortran libopenblas-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
COPY labcoat.yaml /app/labcoat.yaml
COPY glove.yaml /app/glove.yaml
COPY exposed-skin.yaml /app/exposed-skin.yaml
RUN uv sync --frozen

ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
    KAFKA_TOPIC=exposed-skin-alarm \
    KAFKA_GROUP_ID=exposed-skin-alarm-consumer \
    KAFKA_ALARM_TOPIC=exposed-skin-alarm-result \
    KAFKA_BATCH_SIZE=10 \
    KAFKA_MAX_WAIT_SEC=2 \
    KAFKA_MAX_BATCHES=0 \
    KAFKA_AUTO_OFFSET_RESET=latest \
    LABCOAT_MODEL_PATH=/app/model/labcoat/best.pt \
    GLOVE_MODEL_PATH=/app/model/glove/best.pt \
    EXPOSED_SKIN_MODEL_PATH=/app/model/exposed-skin/best.pt \
    LABCOAT_YAML=/app/labcoat.yaml \
    GLOVE_YAML=/app/glove.yaml \
    EXPOSED_SKIN_YAML=/app/exposed-skin.yaml \
    LABCOAT_CLASS_NAME="with labcoat" \
    GLOVE_CLASS_NAME="with glove" \
    EXPOSED_SKIN_CLASS_NAME="exposed-skin" \
    LABCOAT_CLASS_ID= \
    GLOVE_CLASS_ID= \
    EXPOSED_SKIN_CLASS_ID= \
    LABCOAT_CONF=0.7 \
    GLOVE_CONF=0.7 \
    EXPOSED_SKIN_CONF=0.7 \
    MODEL_DEVICE= \
    MODEL_GPU=0 \
    DEBUG_MODE= \
    DEBUG_DIR=/app/kafuka/exposed-skin/debug

CMD ["/bin/sh", "-c", "uv run python /app/kafuka/exposed-skin/exposed_skin_consumer.py \
  --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} \
  --topic ${KAFKA_TOPIC} \
  --group-id ${KAFKA_GROUP_ID} \
  --alarm-topic ${KAFKA_ALARM_TOPIC} \
  --batch-size ${KAFKA_BATCH_SIZE} \
  --max-wait-sec ${KAFKA_MAX_WAIT_SEC} \
  --max-batches ${KAFKA_MAX_BATCHES} \
  --auto-offset-reset ${KAFKA_AUTO_OFFSET_RESET} \
  --labcoat-model ${LABCOAT_MODEL_PATH} \
  --glove-model ${GLOVE_MODEL_PATH} \
  --exposed-skin-model ${EXPOSED_SKIN_MODEL_PATH} \
  --labcoat-yaml ${LABCOAT_YAML} \
  --glove-yaml ${GLOVE_YAML} \
  --exposed-skin-yaml ${EXPOSED_SKIN_YAML} \
  --labcoat-class-name \"${LABCOAT_CLASS_NAME}\" \
  --glove-class-name \"${GLOVE_CLASS_NAME}\" \
  --exposed-skin-class-name \"${EXPOSED_SKIN_CLASS_NAME}\" \
  ${LABCOAT_CLASS_ID:+--labcoat-class-id ${LABCOAT_CLASS_ID}} \
  ${GLOVE_CLASS_ID:+--glove-class-id ${GLOVE_CLASS_ID}} \
  ${EXPOSED_SKIN_CLASS_ID:+--exposed-skin-class-id ${EXPOSED_SKIN_CLASS_ID}} \
  --labcoat-conf ${LABCOAT_CONF} \
  --glove-conf ${GLOVE_CONF} \
  --exposed-skin-conf ${EXPOSED_SKIN_CONF} \
  ${MODEL_DEVICE:+--device ${MODEL_DEVICE}} \
  --gpu ${MODEL_GPU} \
  ${DEBUG_MODE:+--debug} \
  --debug-dir ${DEBUG_DIR}"]

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
RUN uv sync --frozen


ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
    KAFKA_TOPIC=fire-alarm \
    KAFKA_GROUP_ID=fire-alarm-consumer \
    KAFKA_ALARM_TOPIC=fire-alarm-result \
    KAFKA_BATCH_SIZE=10 \
    KAFKA_MAX_WAIT_SEC=2 \
    KAFKA_MAX_BATCHES=0 \
    KAFKA_AUTO_OFFSET_RESET=latest \
    MODEL_PATH=/app/model/fire-kaggle/weights/best.pt \
    MODEL_CONF=0.6 \
    MODEL_DEVICE= \
    MODEL_GPU=0 \
    TARGET_CLASS_NAME=fire \
    TARGET_CLASS_INDEX=0

CMD ["/bin/sh", "-c", "uv run python /app/kafuka/alarm_consumer.py \
  --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} \
  --topic ${KAFKA_TOPIC} \
  --group-id ${KAFKA_GROUP_ID} \
  --alarm-topic ${KAFKA_ALARM_TOPIC} \
  --batch-size ${KAFKA_BATCH_SIZE} \
  --max-wait-sec ${KAFKA_MAX_WAIT_SEC} \
  --max-batches ${KAFKA_MAX_BATCHES} \
  --auto-offset-reset ${KAFKA_AUTO_OFFSET_RESET} \
  --model ${MODEL_PATH} \
  --conf ${MODEL_CONF} \
  ${MODEL_DEVICE:+--device ${MODEL_DEVICE}} \
  --gpu ${MODEL_GPU} \
  --target-class-name ${TARGET_CLASS_NAME} \
  --target-class-index ${TARGET_CLASS_INDEX}"]

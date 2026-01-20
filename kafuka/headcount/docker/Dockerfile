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
    KAFKA_TOPIC=unattended_alarm \
    KAFKA_GROUP_ID=unattended-alarm-consumer \
    KAFKA_ALARM_TOPIC=unattended_alarm_result \
    KAFKA_BATCH_SIZE=10 \
    KAFKA_MAX_WAIT_SEC=2 \
    KAFKA_MAX_BATCHES=0 \
    KAFKA_AUTO_OFFSET_RESET=latest \
    HEADCOUNT_MODEL_PATH=/app/yolo11n.pt \
    HEADCOUNT_CONF=0.5 \
    HEADCOUNT_PERSON_CLASS_NAME=person \
    HEADCOUNT_PERSON_CLASS_ID=0 \
    HEADCOUNT_TIME_START=00:00 \
    HEADCOUNT_TIME_END=08:00 \
    MODEL_DEVICE= \
    MODEL_GPU=0 \
    DEBUG_MODE= \
    DEBUG_DIR=/app/kafuka/headcount/debug

CMD ["/bin/sh", "-c", "uv run python /app/kafuka/headcount/headcount_consumer.py \
  --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} \
  --topic ${KAFKA_TOPIC} \
  --group-id ${KAFKA_GROUP_ID} \
  --alarm-topic ${KAFKA_ALARM_TOPIC} \
  --batch-size ${KAFKA_BATCH_SIZE} \
  --max-wait-sec ${KAFKA_MAX_WAIT_SEC} \
  --max-batches ${KAFKA_MAX_BATCHES} \
  --auto-offset-reset ${KAFKA_AUTO_OFFSET_RESET} \
  --model ${HEADCOUNT_MODEL_PATH} \
  --conf ${HEADCOUNT_CONF} \
  --person-class-name \"${HEADCOUNT_PERSON_CLASS_NAME}\" \
  --person-class-id ${HEADCOUNT_PERSON_CLASS_ID} \
  --time-start ${HEADCOUNT_TIME_START} \
  --time-end ${HEADCOUNT_TIME_END} \
  ${MODEL_DEVICE:+--device ${MODEL_DEVICE}} \
  --gpu ${MODEL_GPU} \
  ${DEBUG_MODE:+--debug} \
  --debug-dir ${DEBUG_DIR}"]

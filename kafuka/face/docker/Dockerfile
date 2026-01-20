FROM python:3.11

ARG http_proxy
ARG https_proxy
ARG all_proxy

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    all_proxy=${all_proxy}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV CMAKE_BUILD_PARALLEL_LEVEL=1 \
    MAKEFLAGS=-j1

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl python3-dev build-essential gfortran libopenblas-dev pkg-config cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
RUN uv sync --frozen

COPY face_consumer.py /app/face_consumer.py
COPY debug_face_check.py /app/debug_face_check.py
COPY face_alarm_producer.py /app/face_alarm_producer.py


ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
    KAFKA_TOPIC=face_recognition_alarm \
    KAFKA_GROUP_ID=face-alarm-consumer \
    KAFKA_ALARM_TOPIC=face_recognition_alarm_result \
    KAFKA_BATCH_SIZE=10 \
    KAFKA_MAX_WAIT_SEC=2 \
    KAFKA_MAX_BATCHES=0 \
    KAFKA_MAX_POLL_INTERVAL_MS=60000 \
    KAFKA_SESSION_TIMEOUT_MS=30000 \
    KAFKA_HEARTBEAT_INTERVAL_MS=10000 \
    KAFKA_AUTO_OFFSET_RESET=latest \
    FACE_PEOPLE_DIR=/app/people \
    FACE_TOLERANCE=0.6 \
    FACE_MODEL=hog \
    FACE_NO_YOLO= \
    FACE_YOLO_MODEL=/app/yolo11s.pt \
    FACE_YOLO_CONF=0.25 \
    FACE_PERSON_CLASS_ID=0 \
    FACE_NUM_UPSAMPLE=0 \
    FACE_GPU= \
    FACE_BATCH_SIZE=128 \
    DEBUG_MODE= \
    DEBUG_DIR=/app/debug

CMD ["/bin/sh", "-c", "APP=/app/face_consumer.py; if [ -f /app/face/face_consumer.py ]; then APP=/app/face/face_consumer.py; fi; \
  uv run python ${APP} \
  --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} \
  --topic ${KAFKA_TOPIC} \
  --group-id ${KAFKA_GROUP_ID} \
  --alarm-topic ${KAFKA_ALARM_TOPIC} \
  --batch-size ${KAFKA_BATCH_SIZE} \
  --max-wait-sec ${KAFKA_MAX_WAIT_SEC} \
  --max-batches ${KAFKA_MAX_BATCHES} \
  --max-poll-interval-ms ${KAFKA_MAX_POLL_INTERVAL_MS} \
  --session-timeout-ms ${KAFKA_SESSION_TIMEOUT_MS} \
  --heartbeat-interval-ms ${KAFKA_HEARTBEAT_INTERVAL_MS} \
  --auto-offset-reset ${KAFKA_AUTO_OFFSET_RESET} \
  --people-dir ${FACE_PEOPLE_DIR} \
  --tolerance ${FACE_TOLERANCE} \
  --model ${FACE_MODEL} \
  ${FACE_NO_YOLO:+--no-yolo} \
  --yolo-model ${FACE_YOLO_MODEL} \
  --yolo-conf ${FACE_YOLO_CONF} \
  --person-class-id ${FACE_PERSON_CLASS_ID} \
  --num-upsample ${FACE_NUM_UPSAMPLE} \
  ${FACE_GPU:+--gpu} \
  --face-batch-size ${FACE_BATCH_SIZE} \
  ${DEBUG_MODE:+--debug} \
  --debug-dir ${DEBUG_DIR}"]

ARG BUILD_FROM
FROM $BUILD_FROM

RUN apk add --no-cache \
    python3 \
    py3-pip \
    ffmpeg \
    jpeg-dev \
    zlib-dev

RUN pip3 install --no-cache-dir --break-system-packages \
    flask \
    yt-dlp \
    opencv-python-headless \
    "numpy<2"

COPY app.py /app.py

CMD [ "python3", "/app.py" ]

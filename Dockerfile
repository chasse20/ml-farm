FROM python:3.12-slim

WORKDIR /app

EXPOSE 8080

ENV SLAVE_ID=0
ENV SLAVE_NAME=
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	git \
	curl \
	unzip \
	wget \
	libboost-all-dev \
	ocl-icd-opencl-dev \
	libopenblas-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
	&& /opt/venv/bin/pip install --upgrade pip setuptools wheel

COPY requirements.txt ./

RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

CMD [ "/opt/venv/bin/python", "Program.py" ]
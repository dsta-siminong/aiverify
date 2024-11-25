FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y python3
RUN apt-get upgrade -y

WORKDIR /workspace

# Install other requirements
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# copy the rest of the files into the container
COPY src .

WORKDIR /workspace/

CMD ["python3", "test.py"]
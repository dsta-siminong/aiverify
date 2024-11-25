FROM ubuntu:22.04

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y python3 \
    && apt-get install python3-pip -y

RUN pip install --upgrade pip
RUN apt-get upgrade -y

WORKDIR /workspace

# Install other requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy the rest of the files into the container
WORKDIR /workspace/src
COPY src .

CMD ["python3", "script.py"]
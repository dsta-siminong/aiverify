# Build the aiverify-test-engine docker image

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

###################  Install libraries ######################

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update 

## Install kubectl
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg

RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
RUN chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg
RUN echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list
RUN chmod 644 /etc/apt/sources.list.d/kubernetes.list
RUN apt-get update
RUN apt-get install -y kubectl


##Install helm
RUN curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | tee /usr/share/keyrings/helm.gpg > /dev/null
RUN apt-get install apt-transport-https --yes
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | tee /etc/apt/sources.list.d/helm-stable-debian.list
RUN apt-get update
RUN apt-get install helm

# Install git
RUN apt-get install -y git

# Install unzip
RUN apt-get install unzip

# Install python 3.11, virtualenv
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11 python3.11-venv

# For shap-toolbox stock plugin
RUN apt-get install -y gcc g++ python3.11-dev

# Install libmagicwand for Environment Corruption
RUN apt-get install libmagickwand-dev -y

# Install OpenGL libraries for OpenCV 
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 

# Set python3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
RUN update-alternatives --set python3 /usr/bin/python3.11

################### Create aiverify user ######################

ARG USER=aiverify
RUN groupadd -g 1000 $USER
RUN useradd -u 1000 -g 1000 -ms /bin/bash $USER

###################  Clone aiverify repo ######################

ARG BRANCH_TAG=main
ARG REPO=imda-btg/aiverify
RUN echo "BRANCH_TAG=$BRANCH_TAG"
WORKDIR /app
RUN git clone https://github.com/$REPO --branch=$BRANCH_TAG aiverify

############### Python #################

# Create file upload and logs folders
WORKDIR /app/aiverify
RUN mkdir -p ./uploads/data
RUN mkdir -p ./uploads/model
RUN mkdir -p ./test-engine-app/logs

RUN chown -R 1000:1000 /app/aiverify

# Install dependencies into virtualenv
WORKDIR /app/aiverify
RUN python3 -m venv venv
ENV PATH="/app/aiverify/venv/bin:$PATH"

# Install dependencies
WORKDIR /app/aiverify/test-engine-app
RUN pip install -r requirements.txt
WORKDIR /app/aiverify/test-engine-core
RUN pip install -r requirements.txt
WORKDIR /app/aiverify/test-engine-core-modules
RUN pip install -r requirements.txt

############## Stock plugins #################
# Unzip stock plugin bundles into the plugins folder
WORKDIR /app/aiverify/stock-plugins/aiverify.stock.shap-toolbox/dist
RUN rm -f aiverify.stock.shap_toolbox-1.0.0.zip
WORKDIR /app/aiverify/stock-plugins
RUN mkdir ../stock-plugins-unzip
RUN for plugin_dir in *; do \
      echo "Unzipping plugin $plugin_dir"; \
      unzip $plugin_dir/dist/*.zip -d ../stock-plugins-unzip/$plugin_dir; \
    done
RUN chown -R 1000:1000 ../stock-plugins-unzip

WORKDIR /app/aiverify/stock-plugins-unzip
RUN find ./ -type f -name 'requirements.txt' -exec pip install -r "{}" \;

WORKDIR /app/aiverify
RUN rm -rf stock-plugins
RUN mv stock-plugins-unzip stock-plugins

WORKDIR /app/aiverify
RUN pip install ./test-engine-core/dist/test_engine_core-*.tar.gz

# Create env file for test-engine-app
WORKDIR /app/aiverify/test-engine-app
RUN echo 'CORE_MODULES_FOLDER="../test-engine-core-modules"\n\
VALIDATION_SCHEMAS_FOLDER="./test_engine_app/validation_schemas/"\n\
REDIS_CONSUMER_GROUP="MyGroup"\n\
REDIS_SERVER_HOSTNAME="redis"\n\
REDIS_SERVER_PORT=6379\n\
API_SERVER_PORT=8080' | tee .env

# Clean up
WORKDIR /app/aiverify
RUN rm -rf ai-verify-portal && rm -rf ai-verify-apigw && rm -rf ai-verify-shared-library
RUN rm -rf third-party && rm -rf setup-aiverify && rm -rf examples && rm -rf scripts && rm -rf deploy-scripts
RUN apt-get remove git unzip -y

# Create plugins folder
RUN mkdir -p ./ai-verify-portal/plugins
RUN chown -R 1000:1000 ./ai-verify-portal

# RUN ls
#RUN pip install deepchecks==0.18.0
# RUN pip install deepchecks[nlp]==0.18.1
# RUN pip install deepchecks[vision]==0.18.1
#RUN pip install torch==2.1.0
#RUN pip install sentence_transformers==2.2.2
#RUN pip install Wand
#RUN apt-get -y install libmagickwand-dev
#RUN pip install opencv-python==4.7.0.72
#RUN pip install lazy_loader==0.3
# RUN pip install adversarial-robustness-toolbox==1.18.0
# RUN pip install pillow==10.4.0
# RUN pip install tensorflow_addons==0.23.0
# RUN pip install minio==7.2.9
# RUN pip install tensorflow[and-cuda]

WORKDIR /app/aiverify/test-engine-app/test_engine_app
# change number of cpu workers so that it doesnt hang
COPY ./test_engine_app.py ./ 

# WORKDIR /app/aiverify/ai-verify-portal/
# COPY ./aiverify-job.yml ./
# feed in task specific inputs here (translate from front end to job parameters)
WORKDIR /app/aiverify/test-engine-app/test_engine_app/processing
COPY ./task_processing.py ./

# for kubectl or helm install
WORKDIR /app/aiverify/test-engine-app
# COPY ./aiverify-job.yml ./
# COPY mychart/ ./mychart/
COPY deepcheckschart/ ./deepcheckschart/
# COPY test_images/ ./test_images/
# COPY xception_test_annotated_labels.csv ./
# COPY Xception-10-0.74.hdf5 ./
# Run containers with non-root user
USER $USER

WORKDIR /app


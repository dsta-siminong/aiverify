Obtaining file:///app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Getting requirements to build editable: started
  Getting requirements to build editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: aiverify-test-engine[all] in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (2.2.0)
Collecting albumentations==2.0.8 (from augmentation_algorithm==0.1.0)
  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
Collecting augly==1.0.0 (from augmentation_algorithm==0.1.0)
  Downloading augly-1.0.0-py3-none-any.whl.metadata (9.4 kB)
Collecting aum==1.0.2 (from augmentation_algorithm==0.1.0)
  Downloading aum-1.0.2-py3-none-any.whl.metadata (315 bytes)
Collecting imagecorruptions==1.1.2 (from augmentation_algorithm==0.1.0)
  Downloading imagecorruptions-1.1.2-py3-none-any.whl.metadata (3.5 kB)
Requirement already satisfied: keras==3.12.0 in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (3.12.0)
Collecting matplotlib==3.10.8 (from augmentation_algorithm==0.1.0)
  Downloading matplotlib-3.10.8-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
Collecting nrtk==0.26.0 (from augmentation_algorithm==0.1.0)
  Downloading nrtk-0.26.0-py3-none-any.whl.metadata (13 kB)
Collecting numpy==1.26.4 (from augmentation_algorithm==0.1.0)
  Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (4.12.0.88)
Requirement already satisfied: pandas==2.2.2 in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (2.2.2)
Requirement already satisfied: pillow==10.4.0 in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (10.4.0)
Collecting plotly==6.5.0 (from augmentation_algorithm==0.1.0)
  Downloading plotly-6.5.0-py3-none-any.whl.metadata (8.5 kB)
Requirement already satisfied: scikit-image==0.25.2 in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (0.25.2)
Requirement already satisfied: scikit-learn==1.5.2 in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (1.5.2)
Requirement already satisfied: scipy==1.14.1 in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (1.14.1)
Requirement already satisfied: torch==2.9.1 in /usr/local/lib/python3.12/site-packages (from augmentation_algorithm==0.1.0) (2.9.1)
Collecting torchvision==0.24.1 (from augmentation_algorithm==0.1.0)
  Downloading torchvision-0.24.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (5.9 kB)
Collecting tqdm==4.67.1 (from augmentation_algorithm==0.1.0)
  Downloading tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Requirement already satisfied: PyYAML in /usr/local/lib/python3.12/site-packages (from albumentations==2.0.8->augmentation_algorithm==0.1.0) (6.0.3)
Requirement already satisfied: pydantic>=2.9.2 in /usr/local/lib/python3.12/site-packages (from albumentations==2.0.8->augmentation_algorithm==0.1.0) (2.12.3)
Collecting albucore==0.0.24 (from albumentations==2.0.8->augmentation_algorithm==0.1.0)
  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
Collecting iopath>=0.1.8 (from augly==1.0.0->augmentation_algorithm==0.1.0)
  Downloading iopath-0.1.10.tar.gz (42 kB)
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
Collecting python-magic>=0.4.22 (from augly==1.0.0->augmentation_algorithm==0.1.0)
  Downloading python_magic-0.4.27-py2.py3-none-any.whl.metadata (5.8 kB)
Collecting regex>=2021.4.4 (from augly==1.0.0->augmentation_algorithm==0.1.0)
  Downloading regex-2025.11.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Collecting opencv-python>=3.4.5 (from imagecorruptions==1.1.2->augmentation_algorithm==0.1.0)
  Downloading opencv_python-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
Requirement already satisfied: absl-py in /usr/local/lib/python3.12/site-packages (from keras==3.12.0->augmentation_algorithm==0.1.0) (2.3.1)
Requirement already satisfied: rich in /usr/local/lib/python3.12/site-packages (from keras==3.12.0->augmentation_algorithm==0.1.0) (14.2.0)
Requirement already satisfied: namex in /usr/local/lib/python3.12/site-packages (from keras==3.12.0->augmentation_algorithm==0.1.0) (0.1.0)
Requirement already satisfied: h5py in /usr/local/lib/python3.12/site-packages (from keras==3.12.0->augmentation_algorithm==0.1.0) (3.15.1)
Requirement already satisfied: optree in /usr/local/lib/python3.12/site-packages (from keras==3.12.0->augmentation_algorithm==0.1.0) (0.17.0)
Requirement already satisfied: ml-dtypes in /usr/local/lib/python3.12/site-packages (from keras==3.12.0->augmentation_algorithm==0.1.0) (0.5.3)
Requirement already satisfied: packaging in /usr/local/lib/python3.12/site-packages (from keras==3.12.0->augmentation_algorithm==0.1.0) (25.0)
Collecting contourpy>=1.0.1 (from matplotlib==3.10.8->augmentation_algorithm==0.1.0)
  Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib==3.10.8->augmentation_algorithm==0.1.0)
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib==3.10.8->augmentation_algorithm==0.1.0)
  Downloading fonttools-4.61.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (114 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib==3.10.8->augmentation_algorithm==0.1.0)
  Downloading kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.3 kB)
Collecting pyparsing>=3 (from matplotlib==3.10.8->augmentation_algorithm==0.1.0)
  Downloading pyparsing-3.3.1-py3-none-any.whl.metadata (5.6 kB)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/site-packages (from matplotlib==3.10.8->augmentation_algorithm==0.1.0) (2.9.0.post0)
Requirement already satisfied: lazy-loader>=0.4 in /usr/local/lib/python3.12/site-packages (from nrtk==0.26.0->augmentation_algorithm==0.1.0) (0.4)
Collecting pycocotools>=2.0.10 (from nrtk==0.26.0->augmentation_algorithm==0.1.0)
  Downloading pycocotools-2.0.11-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (1.3 kB)
Requirement already satisfied: setuptools>=78.1.1 in /usr/local/lib/python3.12/site-packages (from nrtk==0.26.0->augmentation_algorithm==0.1.0) (80.9.0)
Collecting smqtk-classifier>=0.20.0 (from nrtk==0.26.0->augmentation_algorithm==0.1.0)
  Downloading smqtk_classifier-0.20.0-py3-none-any.whl.metadata (2.0 kB)
Collecting smqtk-core>=0.20 (from nrtk==0.26.0->augmentation_algorithm==0.1.0)
  Downloading smqtk_core-0.21.0-py3-none-any.whl.metadata (5.8 kB)
Collecting smqtk-detection>=0.23.0 (from nrtk==0.26.0->augmentation_algorithm==0.1.0)
  Downloading smqtk_detection-0.23.0-py3-none-any.whl.metadata (2.0 kB)
Collecting smqtk-image-io>=0.18.0 (from nrtk==0.26.0->augmentation_algorithm==0.1.0)
  Downloading smqtk_image_io-0.18.0-py3-none-any.whl.metadata (1.6 kB)
Requirement already satisfied: typing-extensions>=4.5.0 in /usr/local/lib/python3.12/site-packages (from nrtk==0.26.0->augmentation_algorithm==0.1.0) (4.15.0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/site-packages (from pandas==2.2.2->augmentation_algorithm==0.1.0) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/site-packages (from pandas==2.2.2->augmentation_algorithm==0.1.0) (2025.2)
Collecting narwhals>=1.15.1 (from plotly==6.5.0->augmentation_algorithm==0.1.0)
  Downloading narwhals-2.14.0-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: networkx>=3.0 in /usr/local/lib/python3.12/site-packages (from scikit-image==0.25.2->augmentation_algorithm==0.1.0) (3.5)
Requirement already satisfied: imageio!=2.35.0,>=2.33 in /usr/local/lib/python3.12/site-packages (from scikit-image==0.25.2->augmentation_algorithm==0.1.0) (2.37.2)
Requirement already satisfied: tifffile>=2022.8.12 in /usr/local/lib/python3.12/site-packages (from scikit-image==0.25.2->augmentation_algorithm==0.1.0) (2025.12.12)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/site-packages (from scikit-learn==1.5.2->augmentation_algorithm==0.1.0) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/site-packages (from scikit-learn==1.5.2->augmentation_algorithm==0.1.0) (3.6.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (3.19.1)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (1.14.0)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (2025.9.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (3.3.20)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (1.13.1.3)
Requirement already satisfied: triton==3.5.1 in /usr/local/lib/python3.12/site-packages (from torch==2.9.1->augmentation_algorithm==0.1.0) (3.5.1)
Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations==2.0.8->augmentation_algorithm==0.1.0)
  Downloading stringzilla-4.6.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations==2.0.8->augmentation_algorithm==0.1.0)
  Downloading simsimd-6.5.12-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
Collecting opencv-python-headless (from augmentation_algorithm==0.1.0)
  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
Requirement already satisfied: aiometer==0.5.0 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.5.0)
Requirement already satisfied: async-timeout==4.0.3 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (4.0.3)
Requirement already satisfied: attrs==23.2.0 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (23.2.0)
Requirement already satisfied: httpx==0.26.0 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.26.0)
Requirement already satisfied: jsonschema-specifications==2023.12.1 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2023.12.1)
Requirement already satisfied: jsonschema==4.21.1 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (4.21.1)
Requirement already satisfied: libclang==16.0.6 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (16.0.6)
Requirement already satisfied: openapi-schema-validator==0.6.2 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.6.2)
Requirement already satisfied: referencing==0.33.0 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.33.0)
Requirement already satisfied: requests>=2.32.4 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2.32.5)
Requirement already satisfied: rpds-py==0.17.1 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.17.1)
Requirement already satisfied: lightgbm>=4.6.0 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (4.6.0)
Requirement already satisfied: tensorflow>=2.18.0 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2.20.0)
Requirement already satisfied: xgboost==2.1.1 in /usr/local/lib/python3.12/site-packages (from aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2.1.1)
Requirement already satisfied: anyio<5,>=3.2 in /usr/local/lib/python3.12/site-packages (from aiometer==0.5.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (4.11.0)
Requirement already satisfied: certifi in /usr/local/lib/python3.12/site-packages (from httpx==0.26.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2025.10.5)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/site-packages (from httpx==0.26.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (1.0.9)
Requirement already satisfied: idna in /usr/local/lib/python3.12/site-packages (from httpx==0.26.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (3.11)
Requirement already satisfied: sniffio in /usr/local/lib/python3.12/site-packages (from httpx==0.26.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (1.3.1)
Requirement already satisfied: rfc3339-validator in /usr/local/lib/python3.12/site-packages (from openapi-schema-validator==0.6.2->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.1.4)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/site-packages (from pydantic>=2.9.2->albumentations==2.0.8->augmentation_algorithm==0.1.0) (0.7.0)
Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/site-packages (from pydantic>=2.9.2->albumentations==2.0.8->augmentation_algorithm==0.1.0) (2.41.4)
Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/site-packages (from pydantic>=2.9.2->albumentations==2.0.8->augmentation_algorithm==0.1.0) (0.4.2)
Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/site-packages (from httpcore==1.*->httpx==0.26.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.16.0)
Collecting portalocker (from iopath>=0.1.8->augly==1.0.0->augmentation_algorithm==0.1.0)
  Downloading portalocker-3.2.0-py3-none-any.whl.metadata (8.7 kB)
INFO: pip is looking at multiple versions of opencv-python to determine which version is compatible with other requirements. This could take a while.
Collecting opencv-python>=3.4.5 (from imagecorruptions==1.1.2->augmentation_algorithm==0.1.0)
  Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib==3.10.8->augmentation_algorithm==0.1.0) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/site-packages (from requests>=2.32.4->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (3.4.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/site-packages (from requests>=2.32.4->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2.5.0)
Collecting smqtk-dataprovider>=0.19.0 (from smqtk-classifier>=0.20.0->nrtk==0.26.0->augmentation_algorithm==0.1.0)
  Downloading smqtk_dataprovider-0.19.0-py3-none-any.whl.metadata (1.9 kB)
Collecting smqtk-descriptors>=0.20 (from smqtk-classifier>=0.20.0->nrtk==0.26.0->augmentation_algorithm==0.1.0)
  Downloading smqtk_descriptors-0.20.0-py3-none-any.whl.metadata (2.1 kB)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/site-packages (from sympy>=1.13.3->torch==2.9.1->augmentation_algorithm==0.1.0) (1.3.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (1.6.3)
Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (25.9.23)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.6.0)
Requirement already satisfied: google_pasta>=0.1.1 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.2.0)
Requirement already satisfied: opt_einsum>=2.3.2 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (3.4.0)
Requirement already satisfied: protobuf>=5.28.0 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (6.33.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (3.2.0)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2.0.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (1.76.0)
Requirement already satisfied: tensorboard~=2.20.0 in /usr/local/lib/python3.12/site-packages (from tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (2.20.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/site-packages (from jinja2->torch==2.9.1->augmentation_algorithm==0.1.0) (2.1.5)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/site-packages (from rich->keras==3.12.0->augmentation_algorithm==0.1.0) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/site-packages (from rich->keras==3.12.0->augmentation_algorithm==0.1.0) (2.19.2)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.12/site-packages (from astunparse>=1.6.0->tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.45.1)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich->keras==3.12.0->augmentation_algorithm==0.1.0) (0.1.2)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.12/site-packages (from tensorboard~=2.20.0->tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (3.9)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.12/site-packages (from tensorboard~=2.20.0->tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.12/site-packages (from tensorboard~=2.20.0->tensorflow>=2.18.0->aiverify-test-engine[all]->augmentation_algorithm==0.1.0) (3.1.3)
Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
Downloading augly-1.0.0-py3-none-any.whl (24.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24.3/24.3 MB 46.6 MB/s eta 0:00:00
Downloading aum-1.0.2-py3-none-any.whl (4.8 kB)
Downloading imagecorruptions-1.1.2-py3-none-any.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 66.5 MB/s eta 0:00:00
Downloading matplotlib-3.10.8-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 69.3 MB/s eta 0:00:00
Downloading nrtk-0.26.0-py3-none-any.whl (153 kB)
Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.0/18.0 MB 71.9 MB/s eta 0:00:00
Downloading plotly-6.5.0-py3-none-any.whl (9.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.9/9.9 MB 72.7 MB/s eta 0:00:00
Downloading torchvision-0.24.1-cp312-cp312-manylinux_2_28_x86_64.whl (8.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.0/8.0 MB 74.0 MB/s eta 0:00:00
Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 61.7 MB/s eta 0:00:00
Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (362 kB)
Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.61.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (5.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.0/5.0 MB 73.3 MB/s eta 0:00:00
Downloading kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 58.8 MB/s eta 0:00:00
Downloading narwhals-2.14.0-py3-none-any.whl (430 kB)
Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (63.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.0/63.0 MB 62.7 MB/s eta 0:00:00
Downloading pycocotools-2.0.11-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (411 kB)
Downloading pyparsing-3.3.1-py3-none-any.whl (121 kB)
Downloading python_magic-0.4.27-py2.py3-none-any.whl (13 kB)
Downloading regex-2025.11.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (803 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 803.5/803.5 kB 101.8 MB/s eta 0:00:00
Downloading smqtk_classifier-0.20.0-py3-none-any.whl (39 kB)
Downloading smqtk_core-0.21.0-py3-none-any.whl (19 kB)
Downloading smqtk_detection-0.23.0-py3-none-any.whl (31 kB)
Downloading smqtk_image_io-0.18.0-py3-none-any.whl (19 kB)
Downloading simsimd-6.5.12-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 583.0/583.0 kB 60.0 MB/s eta 0:00:00
Downloading smqtk_dataprovider-0.19.0-py3-none-any.whl (62 kB)
Downloading smqtk_descriptors-0.20.0-py3-none-any.whl (61 kB)
Downloading stringzilla-4.6.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 76.5 MB/s eta 0:00:00
Downloading portalocker-3.2.0-py3-none-any.whl (22 kB)
Building wheels for collected packages: augmentation_algorithm, iopath
  Building editable for augmentation_algorithm (pyproject.toml): started
  Building editable for augmentation_algorithm (pyproject.toml): finished with status 'done'
  Created wheel for augmentation_algorithm: filename=augmentation_algorithm-0.1.0-0.editable-py3-none-any.whl size=3624 sha256=0db86cc028af6b5fdb38eea33da89e533fb6e3a0b5ec1211590f58f44678e29c
  Stored in directory: /tmp/pip-ephem-wheel-cache-k006jqfd/wheels/29/7c/82/11e078c11cfa1fc194c17fa5512edf2aa240eb555a72eb261f
  Building wheel for iopath (setup.py): started
  Building wheel for iopath (setup.py): finished with status 'done'
  Created wheel for iopath: filename=iopath-0.1.10-py3-none-any.whl size=31600 sha256=f926c317a3f5640f09b904aa180fd8631153395e5ce703f3c1887c49ba64378c
  Stored in directory: /tmp/pip-ephem-wheel-cache-k006jqfd/wheels/7c/96/04/4f5f31ff812f684f69f40cb1634357812220aac58d4698048c
Successfully built augmentation_algorithm iopath
Installing collected packages: simsimd, tqdm, stringzilla, smqtk-core, regex, python-magic, pyparsing, portalocker, numpy, narwhals, kiwisolver, fonttools, cycler, smqtk-dataprovider, pycocotools, plotly, opencv-python-headless, opencv-python, iopath, contourpy, smqtk-image-io, matplotlib, augly, albucore, torchvision, smqtk-descriptors, imagecorruptions, aum, albumentations, smqtk-classifier, smqtk-detection, nrtk, augmentation_algorithm
  Attempting uninstall: numpy
    Found existing installation: numpy 2.2.6
    Not uninstalling numpy at /usr/local/lib/python3.12/site-packages, outside environment /app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm/.venv
    Can't uninstall 'numpy'. No files were found to uninstall.
  Attempting uninstall: opencv-python-headless
    Found existing installation: opencv-python-headless 4.12.0.88
    Not uninstalling opencv-python-headless at /usr/local/lib/python3.12/site-packages, outside environment /app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm/.venv
    Can't uninstall 'opencv-python-headless'. No files were found to uninstall.
  Attempting uninstall: torchvision
    Found existing installation: torchvision 0.24.0
    Not uninstalling torchvision at /usr/local/lib/python3.12/site-packages, outside environment /app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm/.venv
    Can't uninstall 'torchvision'. No files were found to uninstall.
  Attempting uninstall: albumentations
    Found existing installation: albumentations 1.3.0
    Not uninstalling albumentations at /usr/local/lib/python3.12/site-packages, outside environment /app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm/.venv
    Can't uninstall 'albumentations'. No files were found to uninstall.
Successfully installed albucore-0.0.24 albumentations-2.0.8 augly-1.0.0 augmentation_algorithm-0.1.0 aum-1.0.2 contourpy-1.3.3 cycler-0.12.1 fonttools-4.61.1 imagecorruptions-1.1.2 iopath-0.1.10 kiwisolver-1.4.9 matplotlib-3.10.8 narwhals-2.14.0 nrtk-0.26.0 numpy-1.26.4 opencv-python-4.11.0.86 opencv-python-headless-4.11.0.86 plotly-6.5.0 portalocker-3.2.0 pycocotools-2.0.11 pyparsing-3.3.1 python-magic-0.4.27 regex-2025.11.3 simsimd-6.5.12 smqtk-classifier-0.20.0 smqtk-core-0.21.0 smqtk-dataprovider-0.19.0 smqtk-descriptors-0.20.0 smqtk-detection-0.23.0 smqtk-image-io-0.18.0 stringzilla-4.6.0 torchvision-0.24.1 tqdm-4.67.1

[notice] A new release of pip is available: 25.0.1 -> 25.3
[notice] To update, run: python -m pip install --upgrade pip
2026-01-02 06:06:07,439,439 DEBUG    [validate_input.py:26] Validate input data {'aug_library': 'albumentations'}
2026-01-02 06:06:07,440,440 INFO     [virtual_env_execute.py:27] Executing algorithm using venv under /app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm
2026-01-02 06:06:07,440,440 DEBUG    [virtual_env_execute.py:51] cmds: ['/app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm/.venv/bin/python', '-m', 'scripts.algo_execute', '--test_run_id', '46f40938-e984-41c9-9f47-d2809609007a', '--algo_path', '/app/aiverify-test-engine-worker/data/algorithms/cvrob_plugin_augmentation_algorithm', '--data_path', '/app/aiverify-test-engine-worker/data/datasets/all_images_100', '--model_path', '/app/aiverify-test-engine-worker/data/models/ship_pipe', '--model_type', 'classification', '--algorithm_args', '{"aug_library": "albumentations"}', '--apigw_url', 'http://apigw:4000', '--ground_truth_path', '/app/aiverify-test-engine-worker/data/datasets/labels_100.csv', '--ground_truth', 'label']



aiverify-plugin ga augmentation_by_class_algorithm --name "Augmentation by Class Algorithm" --description "This algorithm runs the augmentations and visualizes the trends of metrics by class"
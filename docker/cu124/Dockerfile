FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV CUDA_HOME="/usr/local/cuda"
ENV JAVA_HOME="/usr/lib/jvm/default-java"

ENV PATH=${CUDA_HOME}/bin:${JAVA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

SHELL ["/bin/bash", "-c"]

WORKDIR /root

RUN \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip screen \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-dev \
        libsndfile-dev libcupti-dev libjpeg-dev libpng-dev \
        libaio-dev libnuma-dev && \
        apt-get update && \
        apt-get install -y \
        git python3 python3-pip ninja-build default-jre && \
        python3 -m pip install --upgrade pip wheel && \
        apt-get autoremove -y && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*


# # DOCA https://developer.nvidia.com/doca-archive
# RUN \
#         wget --quiet https://www.mellanox.com/downloads/DOCA/DOCA_v2.5.2/host/doca-host-repo-ubuntu2204_2.5.2-0.0.6.2.5.2003.1.23.10.3.2.2.0_amd64.deb -O /tmp/doca-host-repo-ubuntu2204_2.5.2-0.0.6.2.5.2003.1.23.10.3.2.2.0_amd64.deb && \
#         dpkg -i /tmp/doca-host-repo-ubuntu2204_2.5.2-0.0.6.2.5.2003.1.23.10.3.2.2.0_amd64.deb && \
#         apt-get update && \
#         apt-get -y install doca-runtime doca-sdk doca-tools


# cutlass https://github.com/NVIDIA/cutlass
ENV CUTLASS_PATH="/opt/cutlass"
ENV CUTLASS_NVCC_ARCHS="80;90a"
RUN \
        git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
        cd /opt/cutlass && \
        git fetch --all --tags && \
        git checkout tags/v3.5.0 && \
        git submodule update --init --recursive && \
        export CUDACXX=${CUDA_HOME}/bin/nvcc && \
        mkdir build && \
        cd build && \
        cmake .. -DCUTLASS_NVCC_ARCHS=${CUTLASS_NVCC_ARCHS} -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
        # cmake .. -DCUTLASS_NVCC_ARCHS=${CUTLASS_NVCC_ARCHS} -DCUTLASS_ENABLE_TESTS=ON -DCUTLASS_LIBRARY_KERNELS=all -DCUTLASS_UNITY_BUILD_ENABLED=ON && \
        # make -j"$(nproc)" install
        # make cutlass_profiler -j"$(nproc)"
        # make test_unit -j"$(nproc)" VERBOSE=1


# OPENMPI https://www.open-mpi.org/software/ompi/v4.1/
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.6
RUN \
        wget -q -O /tmp/openmpi-${OPENMPI_VERSION}.tar.gz https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz && \
        tar -xzf /tmp/openmpi-${OPENMPI_VERSION}.tar.gz -C /tmp && \
        cd /tmp/openmpi-${OPENMPI_VERSION} && \
        ./configure --prefix=/opt/openmpi-${OPENMPI_VERSION} && \
        # ./configure --prefix=/opt/openmpi-${OPENMPI_VERSION} --with-cuda=/usr/local/cuda --enable-python-bindings --with-python=/usr/bin/python3 && \
        make -j$(nproc) && \
        make install && \
        ln -s /opt/openmpi-${OPENMPI_VERSION} /opt/openmpi && \
        # Sanity check:
        test -f /opt/openmpi/bin/mpic++ && \
        cd /root && \
        rm -rf /tmp/*
ENV PATH=/opt/openmpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:${LD_LIBRARY_PATH}


# pytorch https://pytorch.org
ENV PYTORCH_VERSION=2.4.1
ENV TORCHVISION_VERSION=0.19.1
ENV TORCHAUDIO_VERSION=2.4.1
ENV PYTORCH_CUDA_VERSION="cu124"
RUN \
        python3 -m pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION} && \
        python3 -m pip install packaging pillow requests jinja2 triton networkx numpy tqdm urllib3 certifi setuptools --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION} && \
        rm -rf /root/.cache/pip


# Install apex with CUDA and C++ extensions https://github.com/NVIDIA/apex
# # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# # otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"
RUN \
        git clone https://github.com/NVIDIA/apex.git /tmp/apex && \
        cd /tmp/apex && \
        git fetch --all --tags && \
        git checkout tags/24.04.01 && \
        git submodule update --init --recursive && \
        python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
        cd /root && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/*


# flash-attention https://github.com/Dao-AILab/flash-attention
# pip install flash-attn --no-build-isolation
# MAX_JOBS=4 pip install flash-attn --no-build-isolation
RUN \
        git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
        cd /tmp/flash-attention && \
        git fetch --all --tags && \
        git checkout tags/v2.6.3 && \
        git submodule update --init --recursive && \
        python3 -m pip install -v -U . && \
        # pytest -q -s tests/test_flash_attn.py
        # cd hopper
        # python3 setup.py install
        # export PYTHONPATH=$PWD
        # pytest -q -s test_flash_attn.py
        cd /root && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/*


# xformers https://github.com/facebookresearch/xformers
RUN \ 
        git clone https://github.com/facebookresearch/xformers.git /tmp/xformers && \
        cd /tmp/xformers && \
        git fetch --all --tags && \
        git checkout tags/v0.0.27.post2 && \
        git submodule update --init --recursive && \
        python3 -m pip install -v -U . && \
        # python3 -m xformers.info && \
        cd /root && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/*


# TransformerEngine https://github.com/NVIDIA/TransformerEngine
RUN \
        git clone https://github.com/NVIDIA/TransformerEngine.git /tmp/TransformerEngine && \
        cd /tmp/TransformerEngine && \
        git fetch --all --tags && \
        git checkout tags/v1.9 && \
        git submodule update --init --recursive && \
        export NVTE_FRAMEWORK=pytorch && \
        export NVTE_WITH_USERBUFFERS=1 && \
        export MPI_HOME=/opt/openmpi && \
        python3 -m pip install -v -U . && \
        cd /root && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/*


RUN \
        python3 -m pip install deepspeed transformers datasets accelerate evaluate peft timm diffusers huggingface_hub trl optimum tokenizers && \
        python3 -m pip install packaging jinja2 triton networkx urllib3 certifi requests protobuf blobfile pytest && \
        python3 -m pip install regex tiktoken sentencepiece tqdm nltk matplotlib seaborn numpy pandas scikit-learn spacy pillow scipy tensorboard && \
        python3 -m pip install pyyaml ipython ipdb pydantic psutil yappi cffi py3nvml pyarrow graphviz astor boto3 msgpack ipykernel cython && \
        rm -rf /root/.cache/pip
RUN \
        python3 -m pip install zstandard nvitop pycocotools tensor_parallel && \
#         # https://github.com/mpi4py/mpi4py/issues/335
#         rm /opt/conda/envs/${CONDA_ENV_NAME}/compiler_compat/ld && \
        python3 -m pip install mpi4py && \
        rm -rf /root/.cache/pip


# Megatron-LM https://github.com/NVIDIA/Megatron-LM
RUN \
        git clone https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM && \
        cd /root/Megatron-LM && \
        git fetch --all --tags && \
        git checkout tags/core_r0.8.0 && \
        git submodule update --init --recursive && \
        python3 -m pip install -e . && \
        cd megatron/core/datasets && \
        make -j$(nproc) && \
        cd /root && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/*


# lm-eval https://github.com/EleutherAI/lm-evaluation-harness
# ENV ANTLR_VERSION=4.13.2
# wget -q -O /root/antlr-${ANTLR_VERSION}-complete.jar https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar
RUN \ 
        apt-get update && \
        apt-get -y install antlr4 && \
        python3 -m pip install immutabledict langdetect && \
        python3 -m nltk.downloader popular punkt punkt_tab && \
        python3 -m pip install antlr4-python3-runtime==4.11 && \
        git clone https://github.com/EleutherAI/lm-evaluation-harness.git /root/lm-evaluation-harness && \
        cd /root/lm-evaluation-harness && \
        git fetch --all --tags && \
        git checkout tags/v0.4.3 && \
        python3 -m pip install -e . && \
        apt-get autoremove -y && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        cd /root && \
        rm -rf /root/.cache/pip && \
        rm -rf /tmp/*


# SSH config
RUN \
        echo 'root:root' | chpasswd && \
        cp /etc/ssh/sshd_config /tmp/sshd_config && \
        echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config && \
        sed -i "s/#Port 22/Port 22124/" /etc/ssh/sshd_config && \
        sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
        sed -i "s/#PasswordAuthentication yes/PasswordAuthentication yes/" /etc/ssh/sshd_config && \
        sed -i "s/#PubkeyAuthentication yes/PubkeyAuthentication yes/" /etc/ssh/sshd_config && \
        sed -i "s/UsePAM yes/UsePAM no/" /etc/ssh/sshd_config && \
        sed -i "s/#StrictModes yes/StrictModes no/" /etc/ssh/sshd_config && \
        sed -i "s/#   StrictHostKeyChecking ask/    StrictHostKeyChecking no/" /etc/ssh/ssh_config && \
        chown root:root /etc/ssh/sshd_config && \
        mkdir -p /run/sshd && chmod 0755 /run/sshd && \
        ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -N "" && \
        cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
# ENV config
RUN \
        unset https_proxy http_proxy && \
        git config --global --unset https.proxy && git config --global --unset http.proxy && \
        echo "CUDA_HOME=${CUDA_HOME}" > ~/.deepspeed_env && \
        echo "CUTLASS_PATH=${CUTLASS_PATH}" >> ~/.deepspeed_env && \
        echo "TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\"" >> ~/.deepspeed_env && \
        echo "PATH=${PATH}" >> ~/.deepspeed_env && \
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ~/.deepspeed_env && \
        echo "LIBRARY_PATH=${LIBRARY_PATH}" >> ~/.deepspeed_env && \
        echo "export CUDA_HOME=${CUDA_HOME}" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export CUTLASS_PATH=${CUTLASS_PATH}" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\"" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export PATH=$PATH" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export LIBRARY_PATH=$LIBRARY_PATH" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc
# clean
RUN \
        cd ~ && \
        apt-get autoremove -y && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        rm -rf /tmp/* && \
        rm -rf /var/tmp/* && \
        rm -rf /root/.cache/pip

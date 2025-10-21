FROM ubuntu:22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ninja-build \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch v0.11.11 --depth 1 https://github.com/sogou/workflow.git && \
    cd workflow && cmake -B cmake-build -G Ninja && cmake --build cmake-build && \
    cp -r ./_lib/* /usr/lib/ && cp -r ./_include/workflow /usr/include/ && \
    cd .. && rm -rf workflow

USER $USERNAME

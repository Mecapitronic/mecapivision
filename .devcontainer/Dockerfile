FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# libgl and x11 for opencv GUI
# nodejs is required for SonarQube
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends \
    git-lfs \
    libgl1 \
    nodejs \
    x11-xserver-utils \
    && apt autopurge -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

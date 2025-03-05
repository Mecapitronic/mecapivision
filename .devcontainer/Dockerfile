FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

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

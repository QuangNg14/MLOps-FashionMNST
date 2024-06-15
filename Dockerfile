# Use an official Debian runtime as a parent image
FROM --platform=linux/amd64 python:3.11-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    gcc \
    python3-dev \
    python3-pip \
    libffi-dev \
    libssl-dev \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

# Install cffi and its dependencies
RUN pip3 install cffi

# Install DVC
RUN pip3 install dvc

# Verify installations
RUN aws --version && dvc --version

# Set the default command to bash
CMD ["bash"]



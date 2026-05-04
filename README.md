<div align="center">
  <h1>🚀 ImageNet Classification Web App</h1>
  <p>
    <b>An enterprise-ready image classification service powered by PyTorch, NVIDIA Triton Inference Server, and Gemini LLM.</b>
  </p>
  <p>
    <a href="https://hub.docker.com/r/yisaienkov/imagenet-classification-web-app"><img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker" alt="Docker Ready"></a>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/NVIDIA_Triton-76B900?logo=nvidia&logoColor=white" alt="NVIDIA Triton">
    <img src="https://img.shields.io/badge/Gemini_API-8E75B2?logo=google&logoColor=white" alt="Gemini">
  </p>
</div>

---

## 📖 Overview

The **ImageNet Classification Web App** is a highly scalable, containerized application designed for real-time image classification. It utilizes a deep learning model to accurately identify the contents of user-uploaded images and enriches the results by generating natural language explanations using Google's **Gemini 1.5 Flash API**. 

The inference is highly optimized, running on **NVIDIA Triton Inference Server** for maximum performance and throughput.

![Application Preview](resources/preview.png)

## ✨ Features

- **High-Performance Inference**: Accelerated model serving using NVIDIA Triton Inference Server with ONNX.
- **AI-Powered Explanations**: Integrates with the Gemini API to provide natural, contextual explanations of the classified objects.
- **Seamless Fallback**: Gracefully falls back to direct PyTorch inference if the Triton server is unreachable.
- **Dockerized Architecture**: Fully containerized using Docker and Docker Compose for simple, reproducible deployments.
- **Interactive UI**: Clean, responsive frontend built with Flask and HTML/CSS.

## 🛠️ Technology Stack

- **Backend Framework**: Flask (Python)
- **Machine Learning**: PyTorch, TorchVision
- **Model Serving**: NVIDIA Triton Inference Server
- **Generative AI**: Google Gemini API
- **Containerization**: Docker, Docker Compose

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your machine:
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- A Gemini API Key (Set in `main.py` or as an environment variable)

### Installation & Deployment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Adithiya-SB/imagenet-classification-web-app.git
   cd imagenet-classification-web-app
   ```

2. **Start the application using Docker Compose:**
   ```bash
   docker-compose up -d --build
   ```
   This command spins up the Flask application and the Triton Inference Server simultaneously.

3. **Access the application:**
   Open your browser and navigate to:
   `http://localhost:5000`

### Local Development (Without Docker)

If you wish to run the app directly on your host machine:

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Start the Flask server:
   ```bash
   python main.py
   ```

## 🧠 Model Pipeline

1. **Preprocessing**: Images are resized to 224x224, converted to tensors, and normalized (ImageNet standards).
2. **Inference**: The input tensor is sent via HTTP client to the Triton Inference Server.
3. **Post-processing**: The model outputs logits, which are converted to probabilities using Softmax to extract the top-3 predictions.
4. **Explanation**: The top class and confidence score are passed to the Gemini LLM to generate a human-readable explanation.

## 📦 Docker Hub

The pre-built image is available on Docker Hub:
[yisaienkov/imagenet-classification-web-app](https://hub.docker.com/r/yisaienkov/imagenet-classification-web-app)

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).


# BBAP-Sec — Real-Time Vision ML Serving Platform

**Production-oriented computer vision inference platform for real-time object detection.**

BBAP-Sec is an extensible ML serving platform built around **FastAPI, YOLO/Ultralytics, Docker, and modern MLOps practices**.

<p align="center">
  <img width="600" alt="ML" src="https://github.com/ties2/Computervision/blob/main/streetAndpeople.jpg">
</p>
---

## Why this project?

A YOLO model alone is not a production ML system.

BBAP-Sec focuses on the engineering layer around the model:

```text
Client
  ↓
FastAPI
  ↓
Inference Engine
  ↓
Model Registry
  ↓
YOLO / Ultralytics
  ↓
Structured Results
```

For real-time workloads:

```text
Camera / RTSP
      ↓
Streaming Pipeline
      ↓
Inference
      ↓
Tracking
      ↓
WebSocket
      ↓
Client
```

---

## Key Features

* REST API for ML inference
* YOLO / Ultralytics model abstraction
* Model registry and version-aware architecture
* Image and real-time video inference
* RTSP / camera streaming support
* WebSocket-based real-time results
* Request validation with Pydantic
* Health and readiness endpoints
* Structured application configuration
* Docker-based deployment
* Automated testing and code quality checks
* MLflow integration planned for experiment tracking and model registry
* DVC integration planned for dataset/version management
* Prometheus + Grafana observability planned

---

## Architecture

```text
                    ┌───────────────┐
                    │    Client     │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │    FastAPI    │
                    │    API Layer  │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │Inference Engine│
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ Model Registry│
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │  Ultralytics  │
                    │     YOLO      │
                    └───────────────┘

RTSP / Camera ──► Streaming Pipeline ──► WebSocket ──► Client

DVC ──► Dataset Versioning
MLflow ──► Experiments / Model Registry
Prometheus ──► Metrics
Grafana ──► Monitoring
Docker ──► Deployment
```

---

## Tech Stack

| Layer               | Technology           |
| ------------------- | -------------------- |
| Language            | Python 3.11+         |
| API                 | FastAPI              |
| ML                  | Ultralytics YOLO     |
| Validation          | Pydantic             |
| Computer Vision     | OpenCV               |
| Testing             | Pytest               |
| Code Quality        | Ruff / MyPy          |
| Containerization    | Docker               |
| Experiment Tracking | MLflow               |
| Data Versioning     | DVC                  |
| Monitoring          | Prometheus / Grafana |

---

## Project Structure

```text
app/
├── api/          # REST/WebSocket API
├── core/         # Configuration, logging, security
├── inference/    # Preprocessing, inference, postprocessing
├── models/       # Model abstraction and registry
└── streaming/    # Camera, RTSP and WebSocket pipeline

configs/          # Environment/model configuration
tests/            # Unit and integration tests
docker/           # Monitoring infrastructure
models/           # Local model artifacts
notebooks/        # Experiments
scripts/          # Utility scripts
doc/              # Architecture and API documentation
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/ties2/Real-Time-Vision-Platform
cd BBAP-Sec
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
make install
```

### 4. Configure environment

```bash
cp .env.example .env
```

### 5. Run

```bash
make run
```

API:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## Development

Run tests:

```bash
make test
```

Run linting:

```bash
make lint
```

Format code:

```bash
make format
```

Run the full quality check:

```bash
make check
```

---

## Roadmap

### Phase 1 — Foundation

* [x] Project architecture
* [x] FastAPI application
* [x] Configuration management
* [x] Logging
* [x] Health/readiness endpoints
* [x] Model abstraction
* [x] Model registry foundation
* [x] Automated tests

### Phase 2 — Model Serving

* [ ] Ultralytics model adapter
* [ ] YOLO11 inference
* [ ] Image inference API
* [ ] Model metadata
* [ ] Batch inference

### Phase 3 — Real-Time Vision

* [ ] Webcam source
* [ ] RTSP source
* [ ] Frame pipeline
* [ ] WebSocket streaming
* [ ] Object tracking
* [ ] FPS / latency optimization

### Phase 4 — MLOps

* [ ] MLflow experiments
* [ ] Model registry
* [ ] DVC datasets
* [ ] Model version management

### Phase 5 — Production

* [ ] Docker Compose
* [ ] Prometheus metrics
* [ ] Grafana dashboards
* [ ] CI/CD
* [ ] Security hardening
* [ ] GPU optimization

---

## Engineering Goals

The long-term goal is not simply to serve a YOLO model.

BBAP-Sec is designed to demonstrate how to turn a computer vision model into a **maintainable, testable, observable, and deployable ML service**.

**Built for real-world computer vision workloads — not just notebooks and demos.**


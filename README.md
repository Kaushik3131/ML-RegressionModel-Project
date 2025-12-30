# 🏠 Housing Price Prediction - End-to-End ML Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-FF6600?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow)
![AWS](https://img.shields.io/badge/AWS-Deployed-FF9900?style=for-the-badge&logo=amazon-aws)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)

An enterprise-grade machine learning pipeline for housing price prediction with automated training, hyperparameter tuning, and production deployment

**🚀 [Live API](#) • [Streamlit Dashboard](#)**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Getting Started](#-getting-started) • [Architecture](#-architecture) • [Deployment](#-deployment)

</div>

---

## 🎯 Overview

Housing Regression MLE is a production-ready end-to-end machine learning system for predicting housing prices using XGBoost. The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization, AWS cloud deployment, and comprehensive testing. The system includes both a REST API and a Streamlit dashboard for interactive predictions.

### Why This Project?

Traditional ML projects often lack production-ready infrastructure. This project demonstrates:

- **End-to-End Pipeline** - From raw data to production deployment
- **MLOps Best Practices** - Experiment tracking, model versioning, automated deployment
- **Data Leakage Prevention** - Time-based splits, proper encoder handling
- **Scalable Architecture** - Containerized microservices on AWS
- **Interactive Dashboards** - Real-time predictions with visual analytics

---

## ✨ Features

### 🤖 Machine Learning Pipeline

- **Automated Feature Engineering** - Date features, frequency encoding, target encoding
- **Time-Aware Data Splitting** - Train (<2020), Eval (2020-21), Holdout (≥2022)
- **Hyperparameter Optimization** - Optuna-based tuning with MLflow tracking
- **Model Versioning** - Automated model saving and loading
- **Batch Predictions** - Monthly prediction generation on holdout data
- **Data Leakage Prevention** - Strict encoder fitting on training data only

### 📊 Interactive Dashboard

- **Real-time Predictions** - Instant housing price predictions via FastAPI
- **Interactive Filtering** - Filter by year, month, and region
- **Visual Analytics** - Predictions vs actuals with MAE, RMSE, % Error
- **Trend Analysis** - Yearly trends with highlighted selected periods
- **Responsive Design** - Optimized for desktop and tablet

### 🚀 REST API

- **Health Checks** - Endpoint monitoring and status
- **Single Predictions** - Real-time prediction endpoint
- **Batch Processing** - Process multiple predictions efficiently
- **S3 Integration** - Automatic model and data loading from AWS S3
- **Error Handling** - Comprehensive error responses

### ☁️ Cloud Infrastructure

- **AWS S3** - Centralized data and model storage
- **Amazon ECR** - Container registry for Docker images
- **Amazon ECS** - Serverless container orchestration with Fargate
- **Application Load Balancer** - Traffic distribution and routing
- **CI/CD Pipeline** - Automated deployment via GitHub Actions

---

## 🛠️ Tech Stack

### Machine Learning

- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting framework
- **[Scikit-learn](https://scikit-learn.org/)** - Feature engineering and preprocessing
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing

### Experiment Tracking & Optimization

- **[MLflow](https://mlflow.org/)** - Experiment tracking and model registry
- **[Optuna](https://optuna.org/)** - Hyperparameter optimization framework

### API & Web Framework

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for APIs
- **[Streamlit](https://streamlit.io/)** - Interactive data science dashboards
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server for FastAPI

### Cloud & DevOps

- **[AWS S3](https://aws.amazon.com/s3/)** - Object storage for data and models
- **[Amazon ECR](https://aws.amazon.com/ecr/)** - Container registry
- **[Amazon ECS](https://aws.amazon.com/ecs/)** - Container orchestration
- **[Docker](https://www.docker.com/)** - Containerization
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD automation

### Development Tools

- **[uv](https://github.com/astral-sh/uv)** - Fast Python package installer
- **[pytest](https://pytest.org/)** - Testing framework
- **[Python 3.11](https://www.python.org/)** - Programming language

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** installed
- **uv** package manager (`pip install uv`)
- **AWS Account** (for deployment)
- **Docker** (for containerization)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Kaushik3131/ML-RegressionModel-Project.git
cd ML-RegressionModel-Project
```

2. **Install dependencies**

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the root directory:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=housing-regression-data

# MLflow Configuration
MLFLOW_TRACKING_URI=./mlruns

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

4. **Download data from S3** (or use local data)

```bash
# Data should be in the data/ directory
# The pipeline will automatically load from S3 if configured
```

### Running Locally

#### 1. Train the Model

```bash
# Basic training
python -m src.training_pipeline.train

# Hyperparameter tuning with Optuna
python -m src.training_pipeline.tune

# Evaluate model
python -m src.training_pipeline.eval
```

#### 2. Start the FastAPI Server

```bash
# Development mode
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 3. Launch Streamlit Dashboard

```bash
streamlit run app.py
```

#### 4. Access the Applications

- **FastAPI API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501

### Available Scripts

```bash
# Training Pipeline
python -m src.training_pipeline.train     # Train baseline model
python -m src.training_pipeline.tune      # Hyperparameter tuning
python -m src.training_pipeline.eval      # Evaluate model

# Inference Pipeline
python -m src.inference_pipeline.inference  # Single prediction
python -m src.batch.run_monthly            # Batch predictions

# API Server
uvicorn src.api.main:app --reload          # Start API server

# Dashboard
streamlit run app.py                       # Start Streamlit app

# Testing
pytest tests/                              # Run all tests
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard          FastAPI REST API              │
│  ├── Interactive Predictions  ├── /predict (single)         │
│  ├── Visual Analytics         ├── /batch (multiple)         │
│  └── Trend Analysis           └── /health (status)          │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────────┬──────────────────┬────────────┐
             ▼                 ▼                  ▼            ▼
    ┌─────────────┐   ┌──────────────┐   ┌──────────┐  ┌─────────┐
    │   AWS S3    │   │   MLflow     │   │ XGBoost  │  │  ECS    │
    │  (Storage)  │   │  (Tracking)  │   │ (Model)  │  │(Deploy) │
    └─────────────┘   └──────────────┘   └──────────┘  └─────────┘
```

### ML Pipeline Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Feature Pipeline                          │
├──────────────────────────────────────────────────────────────┤
│  1. Load Data (load.py)                                      │
│     └── Time-based split: Train <2020, Eval 2020-21, ≥2022  │
│                                                               │
│  2. Preprocess (preprocess.py)                               │
│     ├── City normalization                                   │
│     ├── Deduplication                                        │
│     └── Outlier removal                                      │
│                                                               │
│  3. Feature Engineering (feature_engineering.py)             │
│     ├── Date features (year, month, day)                     │
│     ├── Frequency encoding (zipcode)                         │
│     └── Target encoding (city_full)                          │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   Training Pipeline                          │
├──────────────────────────────────────────────────────────────┤
│  1. Train (train.py)                                         │
│     └── Baseline XGBoost with default params                │
│                                                               │
│  2. Tune (tune.py)                                           │
│     ├── Optuna hyperparameter optimization                   │
│     └── MLflow experiment tracking                           │
│                                                               │
│  3. Evaluate (eval.py)                                       │
│     └── Metrics: MAE, RMSE, R², % Error                     │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  Inference Pipeline                          │
├──────────────────────────────────────────────────────────────┤
│  1. Single Prediction (inference.py)                         │
│     └── Apply saved encoders + model                        │
│                                                               │
│  2. Batch Prediction (run_monthly.py)                        │
│     └── Monthly predictions on holdout data                 │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw Data → Load & Split → Preprocess → Feature Engineering
    ↓
Training Data → XGBoost Training → Model Artifacts
    ↓
Saved Model → Inference Pipeline → Predictions
    ↓
API/Dashboard → User Interface → Visualizations
```

---

## 📁 Project Structure

```
Regression_ML_EndtoEnd/
├── src/
│   ├── feature_pipeline/
│   │   ├── __init__.py
│   │   ├── load.py                  # Time-aware data splitting
│   │   ├── preprocess.py            # Data cleaning & normalization
│   │   └── feature_engineering.py   # Feature creation & encoding
│   ├── training_pipeline/
│   │   ├── __init__.py
│   │   ├── train.py                 # Baseline XGBoost training
│   │   ├── tune.py                  # Optuna hyperparameter tuning
│   │   └── eval.py                  # Model evaluation
│   ├── inference_pipeline/
│   │   ├── __init__.py
│   │   └── inference.py             # Production inference
│   ├── batch/
│   │   └── run_monthly.py           # Batch prediction processing
│   └── api/
│       └── main.py                  # FastAPI REST API
├── app.py                           # Streamlit dashboard
├── data/                            # Local data storage
├── models/                          # Saved models & encoders
├── mlruns/                          # MLflow experiment tracking
├── Dockerfile                       # API container
├── Dockerfile.streamlit             # Streamlit container
├── .github/
│   └── workflows/
│       └── deploy.yml               # CI/CD pipeline
├── pyproject.toml                   # Project dependencies
├── uv.lock                          # Dependency lock file
├── .env                             # Environment variables (gitignored)
└── README.md
```

---

## 🔍 Key Components

### Feature Pipeline

#### 1. Data Loading (`load.py`)

- **Time-based splitting** to prevent data leakage
- Train: < 2020
- Eval: 2020-2021
- Holdout: ≥ 2022

#### 2. Preprocessing (`preprocess.py`)

- City name normalization
- Duplicate removal
- Outlier detection and removal
- Missing value handling

#### 3. Feature Engineering (`feature_engineering.py`)

- **Date features**: Year, month, day extraction
- **Frequency encoding**: Zipcode frequency
- **Target encoding**: City-based target encoding
- Encoder persistence for inference

### Training Pipeline

#### 1. Baseline Training (`train.py`)

- XGBoost regression with default parameters
- Model saving to `models/` directory
- MLflow experiment logging

#### 2. Hyperparameter Tuning (`tune.py`)

- Optuna-based optimization
- MLflow integration for tracking
- Best model selection and saving

#### 3. Model Evaluation (`eval.py`)

- Metrics: MAE, RMSE, R², % Error
- Visualization of predictions vs actuals
- Performance comparison across datasets

### Inference Pipeline

#### Single Prediction (`inference.py`)

- Loads saved model and encoders
- Applies same preprocessing transformations
- Returns prediction with confidence

#### Batch Processing (`run_monthly.py`)

- Processes monthly data in batches
- Generates predictions for holdout set
- Saves results to S3

### API & Dashboard

#### FastAPI (`src/api/main.py`)

- **Endpoints**:
  - `GET /health` - Health check
  - `POST /predict` - Single prediction
  - `POST /batch` - Batch predictions
- S3 integration for model loading
- Comprehensive error handling

#### Streamlit (`app.py`)

- Interactive prediction interface
- Real-time API integration
- Visual analytics with charts
- Filtering by year, month, region

---

## 🔐 Data Leakage Prevention

The project implements strict measures to prevent data leakage:

### 1. Time-Based Splits

- **No random splitting** - Uses chronological order
- Train: Historical data (< 2020)
- Eval: Recent data (2020-2021)
- Holdout: Future data (≥ 2022)

### 2. Encoder Handling

- Encoders fitted **only on training data**
- Same encoders applied to eval/holdout/inference
- Encoder artifacts saved and versioned

### 3. Feature Engineering

- Target encoding uses only training statistics
- No future information leakage
- Proper handling of unseen categories

### 4. Schema Alignment

- Consistent feature sets across all stages
- Validation of input data schema
- Automatic handling of missing features

---

## 📊 Model Performance

### Metrics

| Metric | Train | Eval | Holdout |
|--------|-------|------|---------|
| MAE    | TBD   | TBD  | TBD     |
| RMSE   | TBD   | TBD  | TBD     |
| R²     | TBD   | TBD  | TBD     |
| % Error| TBD   | TBD  | TBD     |

### Feature Importance

Top features contributing to predictions:
- Date features (year, month)
- Location (city, zipcode)
- Property characteristics
- Encoded categorical features

---

## 🚢 Deployment

### Docker Deployment

#### Build Images

```bash
# Build API image
docker build -t housing-api -f Dockerfile .

# Build Streamlit image
docker build -t housing-streamlit -f Dockerfile.streamlit .
```

#### Run Containers

```bash
# Run API
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  housing-api

# Run Streamlit
docker run -p 8501:8501 \
  -e API_URL=http://api:8000 \
  housing-streamlit
```

### AWS ECS Deployment

#### Prerequisites

- AWS CLI configured
- ECR repository created
- ECS cluster set up

#### Deployment Steps

1. **Push to ECR**

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag images
docker tag housing-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/housing-api:latest
docker tag housing-streamlit:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/housing-streamlit:latest

# Push images
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/housing-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/housing-streamlit:latest
```

2. **Deploy to ECS**

```bash
# Update ECS services (via GitHub Actions or manually)
aws ecs update-service --cluster housing-cluster --service housing-api-service --force-new-deployment
aws ecs update-service --cluster housing-cluster --service housing-streamlit-service --force-new-deployment
```

### Environment Variables Setup

Required environment variables for deployment:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_REGION=us-east-1
S3_BUCKET_NAME=housing-regression-data

# MLflow
MLFLOW_TRACKING_URI=./mlruns

# API
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit
STREAMLIT_SERVER_PORT=8501
API_URL=http://api:8000
```

### CI/CD Pipeline

The project uses GitHub Actions for automated deployment:

1. **Trigger**: Push to `main` branch
2. **Build**: Docker images built
3. **Test**: Run pytest suite
4. **Push**: Images pushed to ECR
5. **Deploy**: ECS services updated

---

## 🧪 Testing

### Manual Testing Checklist

- [ ] Feature pipeline runs without errors
- [ ] Training pipeline produces valid model
- [ ] Inference pipeline returns predictions
- [ ] API endpoints respond correctly
- [ ] Streamlit dashboard loads and functions
- [ ] Docker containers build and run
- [ ] AWS deployment successful

### Future Testing Plans

- Unit tests for each pipeline component
- Integration tests for API endpoints
- End-to-end tests for full pipeline
- Performance benchmarking
- Load testing for API

---

## 🎯 Roadmap

### Phase 1: Core Features ✅ (Completed)

- [x] Feature engineering pipeline
- [x] XGBoost training pipeline
- [x] Hyperparameter tuning with Optuna
- [x] MLflow experiment tracking
- [x] FastAPI REST API
- [x] Streamlit dashboard
- [x] Docker containerization
- [x] AWS ECS deployment

### Phase 2: Enhancements 🚧 (In Progress)

- [ ] Automated testing suite
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Advanced feature engineering
- [ ] Model explainability (SHAP values)

### Phase 3: Advanced Features 📋 (Planned)

- [ ] Real-time model retraining
- [ ] Multi-model ensemble
- [ ] AutoML integration
- [ ] Advanced visualization dashboard
- [ ] API rate limiting and caching

### Phase 4: Scaling 🔮 (Future)

- [ ] Kubernetes deployment
- [ ] Distributed training
- [ ] Feature store integration
- [ ] Model serving optimization
- [ ] Multi-region deployment

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features
- Update documentation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Kaushik**

- GitHub: [@Kaushik3131](https://github.com/Kaushik3131)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- **XGBoost Team** - For the powerful gradient boosting framework
- **MLflow** - For experiment tracking capabilities
- **FastAPI** - For the modern API framework
- **Streamlit** - For the interactive dashboard framework
- **AWS** - For cloud infrastructure

---

## 📧 Support

For questions or support, please:

- Open an issue on GitHub
- Contact via email: your.email@example.com
- Join our Discord community: [Link]

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Kaushik

</div>

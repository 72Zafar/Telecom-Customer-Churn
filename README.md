# Telecom-Customer-Churn-Prediction


# **🚀 End-to-End MLOps Project – Production-Grade ML Pipeline**

This repository contains a fully modular, enterprise-grade **MLOps architecture** demonstrating industry-standard practices for building, training, deploying, and maintaining Machine Learning systems.

The project implements a complete ML lifecycle including:

* 🧱 Project Scaffolding & Environment Setup
* 🍃 **MongoDB Atlas** for cloud-hosted data storage
* 📑 Logging, Exception Handling & Notebooks
* 📥 Data Ingestion
* 🛡️ Data Validation
* 🔄 Data Transformation
* 🤖 Model Training
* 📊 Model Evaluation & Model Registry on **AWS S3**
* 🚀 Model Deployment with **Docker + ECR + EC2 + GitHub Actions CI/CD**
* 📡 Prediction Pipeline + Web App

This README walks through the full workflow and highlights the tools & engineering practices used — ideal for hiring managers reviewing the project.

---

# **📂 1. Project Setup & Environment**

### **1.1 Generate Project Template**

```bash
python template.py
```

### **1.2 Add Local Package Configuration**

Configured using:

* `setup.py`
* `pyproject.toml`

(Refer to crashcourse.txt for deep explanations.)

### **1.3 Create Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify local packages:

```bash
pip list
```

---

# **🍃 2. MongoDB Atlas Setup**

### **Atlas Cluster Setup**

1. Sign in → Create new project
2. Create free cluster → **M0** tier
3. Create DB User (username + password)
4. Allow global access:
   `0.0.0.0/0`
5. Get Connection String:

   * Driver: Python
   * Version: 3.6+
6. Store connection string for pipeline configuration.

### **Notebook Setup**

* Create folder: `notebook/`
* Add your dataset
* Create `mongoDB_demo.ipynb`
* Push dataset to MongoDB
* Validate in: **Atlas → Database → Collections**

---

# **🧾 3. Logging, Exception Handling & Notebooks**

* `logger.py` created and validated via `demo.py`
* `exception.py` created and validated via `demo.py`
* EDA Notebook
* Feature Engineering Notebook

---

# **📥 4. Data Ingestion Module**

Before implementation:

### **4.1 Update Configuration Files**

* `constants.__init__.py`
* `configuration.mongo_db_connections.py`
* `data_access/proj1_data.py`
* `entity.config_entity.py`
* `entity.artifact_entity.py`

### **4.2 Build Data Ingestion Component**

* Implement logic in `components.data_ingestion.py`
* Add ingestion stage to training pipeline
* Run using:

```bash
python demo.py
```

### **4.3 Configure Environment Variable**

**Mac/Linux Bash**

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>..."
```

**Windows PowerShell**

```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>..."
```

Also add `artifact/` folder to `.gitignore`.

---

# **🛡️ 5. Data Validation, Transformation & Model Training**

### **5.1 Data Validation**

* Update `utils.main_utils.py`
* Define schema in `config.schema.yaml`
* Create validation component similar to ingestion module

### **5.2 Data Transformation**

* Implement transformer logic
* Add `estimator.py` under `entity/`

### **5.3 Model Trainer**

* Add training logic
* Add model class in `estimator.py`

---

# **☁️ 6. AWS Integration (Model Evaluation & Pusher)**

### **AWS Services Used**

* **IAM**
* **S3**
* **ECR**
* **EC2**

### **6.1 Create IAM User**

* Region: `us-east-1`
* Policy: `AdministratorAccess`
* Create Access Key for CLI

### **6.2 Configure Credentials (Bash)**

```bash
export AWS_ACCESS_KEY_ID="YOUR_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET"
```

### **6.3 Configure (PowerShell)**

```powershell
$env:AWS_ACCESS_KEY_ID="YOUR_KEY"
$env:AWS_SECRET_ACCESS_KEY="YOUR_SECRET"
```

### **6.4 Update constants**

```
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
```

### **6.5 Create S3 Bucket**

* Name: `my-model-mlopsproj`
* Region: `us-east-1`
* Public Access: Disabled (except ACL)

### **6.6 Implement AWS Storage Layer**

* Add AWS-related functions in `src/aws_storage`
* Add S3 Estimator logic in `entity/s3_estimator.py`

---

# **🚀 7. Model Evaluation & Pusher Components**

After completing AWS setup:

* Build Model Evaluation
* Build Model Pusher
* Link both to pipeline

Commit:

```
git commit -m "Added Model Evaluation and Model Pusher components"
```

---

# **🧮 8. Prediction Pipeline + Web Application**

### **8.1 Build Prediction Pipeline**

* Create prediction modules
* Prepare `app.py`

### **8.2 Add UI Assets**

* Add `static/`
* Add `templates/`

---

# **🔁 9. CI/CD Pipeline with GitHub Actions, Docker, ECR & EC2**

### **9.1 Docker Setup**

* Create `Dockerfile`
* Create `.dockerignore`

### **9.2 GitHub Actions Workflow**

* Create `.github/workflows/aws.yaml`

### **9.3 Create New IAM User: `usvisa-user`**

* Generate Access Key
* Add to GitHub Secrets:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
```

### **9.4 Create ECR Repository**

* Name: `vehicleproj`
* Region: `us-east-1`

### **9.5 Setup EC2 Instance**

* Ubuntu 24.04
* t2.medium
* Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### **9.6 Connect GitHub Self-Hosted Runner**

Follow GitHub’s auto-generated commands on EC2:

* Download Runner
* Configure Runner
* Start Runner using:

```bash
./run.sh
```

---

# **🌐 10. Deployment**

### **Expose Port on EC2**

EC2 → Security Group → Inbound Rules:

```
Type: Custom TCP
Port: 5080
Source: 0.0.0.0/0
```

### **Access Application**

```
http://<EC2-PUBLIC-IP>:5080
```

### **Train Model from Browser**

```
/training
```

---

# **🧱 Project Architecture**

```
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── entity/
│   ├── utils/
│   ├── configuration/
│   ├── data_access/
│   └── aws_storage/
├── app.py
├── setup.py
├── pyproject.toml
├── requirements.txt
├── notebook/
├── static/
├── templates/
└── artifact/
```

---

# **🛠️ Tech Stack**

### **Languages & Frameworks**

* Python
* FastAPI / Flask
* Scikit-learn
* Pandas, NumPy

### **MLOps & Cloud**

* MongoDB Atlas
* AWS S3, ECR, EC2, IAM
* Docker
* GitHub Actions (CI/CD)

---

# **📌 Summary**

This project demonstrates **end-to-end MLOps engineering** including:

✔ Modular & scalable codebase
✔ Clear configuration-driven architecture
✔ Cloud-first design (MongoDB + AWS)
✔ Automated CI/CD deployment
✔ Real-time prediction service

Perfect for showcasing **production-level ML engineering skills** to recruiters and technical interviewers.


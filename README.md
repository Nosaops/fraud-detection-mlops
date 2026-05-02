# Fraud Detection API

A machine learning API that predicts fraudulent transactions, built to deepen my understanding of ML integration in microservices architecture as an Operations and Product Manager.

---

## Background

I led a team that built a fraud detection dashboard for a thrift finance firm dealing with internal fund manipulation. The product worked  but I wanted more to understand the engineering layer beneath it as i transition into technical product management.

---

## What It Does

- Accepts transaction data as a JSON payload via a REST endpoint
- Runs it through a pre-trained Random Forest classifier
- Returns a fraud prediction with a confidence score
- Packaged as a Docker container with a CI/CD pipeline via GitHub Actions

---

## Stack

| | |
|---|---|
| Language | Python |
| API | Flask |
| Model | Scikit-learn (Random Forest + GridSearchCV) |
| Container | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## Running Locally

```bash
git clone https://github.com/Nosaops/fraud-detection-mlops.git
cd fraud-detection-mlops

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python fraudapp.py
```

### With Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

---

## API

**`POST /predict`**

```json
Server test A

http://localhost:8000/docs

JSON File
{
  "Time": 0.0,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  "V4": 1.37815522427443,
  "V5": -0.338320769942518,
  "V6": 0.462387777762292,
  "V7": 0.239598554061257,
  "V8": 0.0986979012610507,
  "V9": 0.363786969611213,
  "V10": 0.0907941719789316,
  "V11": -0.551599533260813,
  "V12": -0.617800855762348,
  "V13": -0.991389847235408,
  "V14": -0.311169353699879,
  "V15": 1.46817697209427,
  "V16": -0.470400525259478,
  "V17": 0.207971241929242,
  "V18": 0.0257905801985591,
  "V19": 0.403992960255733,
  "V20": 0.251412098239705,
  "V21": -0.018306777944153,
  "V22": 0.277837575558899,
  "V23": -0.110473910188767,
  "V24": 0.0669280749146731,
  "V25": 0.128539358273528,
  "V26": -0.189114843888824,
  "V27": 0.133558376740387,
  "V28": -0.0210530534538215,
  "Amount": 149.62
}

Server test B

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0, "V1": -1.35, "V2": -0.07, "V3": 2.53,
    "V4": 1.37, "V5": -0.33, "V6": 0.46, "V7": 0.23,
    "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55,
    "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46,
    "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40,
    "V20": 0.25, "V21": -0.01, "V22": 0.27, "V23": -0.11,
    "V24": 0.06, "V25": 0.12, "V26": -0.18, "V27": 0.13,
    "V28": -0.02, "Amount": 149.62
  }'


// Response
{
  "prediction": "FRAUD",
  "confidence": 0.87
}
```

---

## Container Running

The API containerized and running in Docker Desktop, exposed on port `8000:8000`:

![fraud-api container running in Docker Desktop](./Pictures/Screenshot_2026-05-02_132547.png)
> CPU usage: 0.25% — lightweight enough to sit alongside other microservices in a finance system.

---

## Tests

```bash
pytest tests/
```

---

## Project Structure

```
fraud-detection-mlops/
├── fraudapp.py          # Flask API
├── gs_rf.pkl            # Trained model
├── check_model.py       # Model inspection script
├── Dockerfile
├── requirements.txt
├── tests/
└── .github/workflows/   # CI/CD pipeline
```

---

## About

Built by EMMANUEL NOSAKHARE ASOWATA **Nosaops** — An Operations Manager and Product enthusiast transitioning into Technical product management.

See my other projects → [My Portfolio on Notion](https://www.notion.so/My-Portfolio-3354106e55f7800d85c2e3052006172d)

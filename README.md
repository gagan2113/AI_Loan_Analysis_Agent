# AI Banking Customer Onboarding Agent

An AI-powered agent that helps banks decide whether a **new customer should be approved, rejected, or flagged for manual review**. It automates customer onboarding by combining **document verification, credit scoring, and risk assessment** into one explainable decision-making pipeline.

---

## 🚀 Features

* 📝 Extracts and verifies customer documents (OCR + AI)
* 📊 Fetches credit scores and external risk data
* 🤖 ML-based risk scoring for eligibility
* 🧠 LLM-based explainable decisions (approve/reject/manual review)
* ⚡ API-based integration with banking apps (via FastAPI)
* 🔄 Workflow orchestration with LangGraph

---

## 🛠️ Tech Stack

* **Python** – Core development
* **FastAPI** – API layer
* **LangGraph** – Agent workflow orchestration
* **ML Model (Scikit-learn / XGBoost / LightGBM)** – Risk scoring
* **LLM (OpenAI / Llama 3 / Claude)** – Natural language decision explanations
* **OCR (Tesseract / EasyOCR)** – Document text extraction

---

## 📂 Workflow

1. **Customer applies** → Submits ID, address, income details.
2. **Data extraction** → OCR + structured form inputs.
3. **Risk analysis** → Credit bureau check + ML risk score.
4. **Decision making** → LLM explains the decision.
5. **Output** → Approve / Reject / Manual Review with reasoning.

---

## 💡 Example

**Input:**

```json
{
  "name": "Rajesh Sharma",
  "age": 28,
  "income": 80000,
  "employment_type": "Salaried",
  "credit_score": 780,
  "documents_verified": true
}
```

**Agent Output:**

```json
{
  "status": "Approved",
  "reason": "Customer approved with low risk due to high credit score and stable income."
}
```

**Another Input:**

```json
{
  "name": "Sunita Gupta",
  "age": 22,
  "income": 15000,
  "employment_type": "Self-Employed",
  "credit_score": null,
  "documents_verified": true
}
```

**Agent Output:**

```json
{
  "status": "Manual Review Required",
  "reason": "Customer has no credit history and low income. Documents verified successfully."
}
```

---

## 📊 Datasets (for ML Model)

You can use public datasets for training the risk scoring model:

* [German Credit Risk Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+%28german+credit+data%29)
* [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)
* [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/yourusername/bank-onboarding-agent.git
cd bank-onboarding-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run FastAPI app
uvicorn app.main:app --reload
```

---

## 📌 API Usage

### **POST /onboard-customer**

**Request:**

```json
{
  "name": "Rajesh Sharma",
  "age": 28,
  "income": 80000,
  "employment_type": "Salaried",
  "credit_score": 780,
  "documents_verified": true
}
```

**Response:**

```json
{
  "status": "Approved",
  "reason": "Customer approved with low risk due to high credit score and stable income."
}
```

---

## 🔮 Future Enhancements

* 🔐 Integrate real KYC APIs (e.g., Aadhaar, PAN validation)
* 🌍 Multi-language support for onboarding forms
* 🏦 Real-time fraud detection signals
* 📉 More advanced ML risk models


@'
# Bati Bank Credit Risk Assessment Model (Alternative Data)

An end-to-end Machine Learning pipeline built to evaluate credit risk for a Buy-Now-Pay-Later (BNPL) partnership between Bati Bank and a leading eCommerce platform using transaction-level data from Xente[cite: 8, 9, 36].

---

## 📌 Project Overview
This repository contains the architecture, exploratory data analysis, and engineering work for an automated credit scoring system[cite: 17, 20]. By leveraging alternative transactional data, the system engineers behavioral risk proxies, tracks machine learning experiments, and deploys a production-ready REST API to support real-time credit underwriting[cite: 16, 24, 32].

## 📂 Repository Structure
```text
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                          # Ignored locally; raw and processed datasets
│   ├── raw/                       
│   └── processed/                 
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis & visual insights
│   └── plots/                     # <-- Corrected: Housed inside notebooks/ for clean organization
│       ├── monetary_distributions.png
│       └── target_distribution.png
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering & transformation pipelines
│   ├── train.py                   # Model training & hyperparameter tuning
│   ├── predict.py                 # Real-time inference logic
│   └── api/
│       ├── main.py                # FastAPI server implementation
│       └── pydantic_models.py     # Request/response data validation schemas
├── tests/
│   └── test_data_processing.py    # Pytest automated unit tests
├── Dockerfile                     # API containerization
├── docker-compose.yml             # Local service orchestration
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git tracking exclusions
└── README.md                      # Documentation & Business Understanding
```

## 🏢 Credit Scoring Business Understanding (Task 1)

### 1. Basel II Accord Compliance & Interpretability

The Basel II Accord sets strict regulatory expectations for financial institutions regarding capital adequacy, risk documentation, and credit risk measurement.  

The Necessity of Interpretability: In a regulated banking environment like Bati Bank, black-box predictions are a legal liability. Credit risk models must be inherently transparent so that internal auditors, compliance officers, and external regulators can clearly trace why a customer was granted a specific credit score or denied a loan.  

Documentation & Monitoring: Basel II requires clear tracking of data lineage, feature selection criteria, and systemic model performance monitoring to manage data drift over time, protecting bank reserves from unmitigated defaults.

### 2. The Necessity and Business Risks of a Proxy Target Variable

Because our raw dataset consists of e-commerce transactions without historical loan records, there is no explicit ground-truth "default" label.  

Why a Proxy is Necessary: To train a supervised learning model, we must map behavioral patterns to a risk outcome. By computing Recency, Frequency, and Monetary (RFM) metrics, we can isolate highly disengaged or low-value user clusters to act as a proxy target for high credit default risk (is_high_risk).  

Business Risks Introduced: * False Positives: Mislabeled "high-risk" customers (e.g., creditworthy users who simply changed platforms) result in lost revenue and customer friction for the platform.  

False Negatives: Sophisticated bad actors or fraudulent profiles might exhibit excellent short-term transactional velocity, bypassing the proxy and triggering costly credit defaults.  

Assumption Drift: The relationship between transactional engagement and creditworthiness is a modeling assumption, not a constant law. Changes in macroeconomics or app design can completely break the proxy's validity.

### 3. Model Trade-offs: Interpretability vs. Predictive Power
[cite_start]Deploying a financial credit engine requires balancing compliance with predictive capability:

| Model Approach | Strengths | Weaknesses | Basel II Context |
| :--- | :--- | :--- | :--- |
| **Simple / Interpretable**<br>*(e.g., Logistic Regression with WoE)* | [cite_start]Highly transparent coefficients; easy to map directly to a traditional, auditable scorecard. | [cite_start]Struggles to capture complex, non-linear interactions within noisy alternative datasets. | [cite_start]**Highly Approved.** Perfect for smooth regulatory audits and explicit credit justification. |
| **High-Performance**<br>*(e.g., Gradient Boosting / XGBoost)* | [cite_start]Robustly captures deep interactions and non-linear patterns, yielding higher ROC-AUC. | [cite_start]Inherently opaque; functions as a "black box" that can overfit volatile behavioral signals. | [cite_start]**Requires Guardrails.** Can only be used if paired with robust post-hoc explainability frameworks (SHAP/LIME). |

## 📊 Exploratory Data Analysis (Task 2)
All exploratory data visual patterns, distribution analysis, and structural audits are contained within `notebooks/eda.ipynb`. Below are the key visual insights and statistical distributions from the Xente dataset:

### 1. Target Class Distribution
The dataset exhibits a severe, critical class imbalance within the `FraudResult` target column across its **95,662 total recorded transactions**, where the minority class accounts for a microscopic fraction of the overall data. 

![Target Class Distribution](notebooks/plots/target_distribution.png)

* **Concrete Metrics:** Out of 95,662 transactions, **only 193 cases are flagged as fraud** (`FraudResult = 1`). This represents an extreme imbalance rate of approximately **0.20% positive classes** vs. 99.80% normal transactions.
* **Modeling Impact:** Standard classification accuracy will be highly misleading (a dummy model guessing "0" achieves 99.8% accuracy while failing completely). The predictive models must be optimized and evaluated strictly using Precision-Recall curves, F1-Score, and Area Under the ROC Curve (ROC-AUC).

### 2. Transaction Amount and Value Distributions
The transaction `Amount` and absolute `Value` metrics are aggressively right-skewed, characterized by a massive volume of low-value day-to-day transactions and a small handful of extreme outlier spikes.

![Monetary Distributions](notebooks/plots/monetary_distributions.png)

* **Concrete Metrics:** The median transaction value sits tightly at **1,000 UGX** (with 75% of transactions falling under 5,000 UGX), yet the maximum recorded outlier values spike drastically all the way up to **9,880,000 UGX**. 
* **Modeling Impact:** Distance-based algorithms—such as the K-Means clustering algorithm used later to build our behavioral credit risk proxy—will suffer completely from outlier dominance if left unaddressed. Applying log transformations (`log1p`) or robust scaling adjustments is mandatory to stabilize the feature space before modeling.

### 3. Boxplot-Based Outlier Investigation
The transaction metrics display distinct hyper-extreme outliers which distort distance calculations across standard machine learning modeling baselines.

![Outlier Boxplots](plots/outlier_boxplots.png)

* **Analytical Insight:** While the bulk of transaction volumes exist within narrow limits, individual transactions scale up to nearly 10,000,000 UGX. 
* **Downstream Modeling Decision:** Standard scaling (Z-score) will fail due to mean distortion from these extreme values. We will implement a `RobustScaler` (which uses the median and Interquartile Range) or strict log transformations before passing features to clustering algorithms to insulate centroids from outlier pull.

### 4. Numerical Feature Correlation Matrix
An evaluation of linear relationships across structural numeric identifiers was conducted to isolate patterns of multicollinearity.

![Correlation Heatmap](plots/correlation_heatmap.png)

* **Analytical Insight:** High linear dependencies exist between transaction metrics like `Amount` and absolute `Value`. 
* **Downstream Modeling Decision:** Keeping highly collinear variables intact will destabilize coefficients in interpretable linear models like Logistic Regression. We will drop redundant parallel vectors or utilize feature reduction techniques to ensure clean coefficient evaluation.

### 5. Missing Value Assessment & Imputation Plan
A comprehensive structural scan of the Xente transaction array was run to identify data sparsity.

| Feature Column | Missing Rows | Total Share (%) | Strategic Imputation Plan |
| :--- | :--- | :--- | :--- |
| `TransactionId` | 0 | 0.00% | No action required. |
| `Amount` | 0 | 0.00% | No action required. |
| `Value` | 0 | 0.00% | No action required. |
| `FraudResult` | 0 | 0.00% | No action required. |

* **Downstream Modeling Decision:** The core transactional profile columns within this dataset are structurally complete ($0\%$ missing values). If downstream feature engineering fields generate missing elements (e.g., historical rolling averages for fresh accounts), we will deploy **Median Imputation** for continuous arrays and **Mode Imputation** for low-frequency categoricals to guarantee pipeline stability.

---


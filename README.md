# Self-Healing Machine Learning Pipeline

### An Automated, Context-Aware Data Cleaning & Repair System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

## 📖 Overview

The **Self-Healing ML Pipeline** is an intelligent data preprocessing tool designed to turn "messy" raw datasets into machine-learning-ready data without manual intervention. 

Unlike standard cleaning scripts that drop missing rows or fill them with static means, this pipeline uses **Machine Learning to fix Machine Learning**. It employs an ensemble of Random Forests to predict missing values based on hidden patterns in the data and Isolation Forests to surgically remove statistical anomalies while preserving valid edge cases.

## ✨ Key Features

### 1. Universal Data Repair (The "Universal Translator")
* **Domain Agnostic:** Works on any CSV dataset (HR, Finance, Medical, etc.) without hardcoded rules.
* **Categorical Handling:** Automatically detects text columns (e.g., "Department", "Gender"), converts them using **Ordinal Encoding**, repairs them, and decodes them back to the original text.
* **Dimension-Safe:** Handles high-cardinality features and manages 1D/2D array transformations seamlessly.

### 2. Context-Aware Imputation
* **Algorithm:** `IterativeImputer` backed by a `RandomForestRegressor`.
* **Logic:** Instead of filling `NaN` with the average (which distorts data), the pipeline treats every missing value as a target variable. It uses valid features (e.g., *Experience, Department*) to predict the missing feature (e.g., *Salary*) with high precision.

### 3. Explainable Outlier Detection
* **Algorithm:** `IsolationForest` (Unsupervised Anomaly Detection).
* **Dynamic Reasoning:** The "Black Box" problem is solved by a custom reasoning engine. It calculates Z-scores and probability densities to tell the user *why* a row was flagged:
    * *"Salary is extremely high"* (Magnitude Outlier)
    * *"Rare value in 'Department'"* (Categorical Rarity)
    * *"Unusual combination of Age and Experience"* (Contextual Mismatch)

### 4. Human-in-the-Loop Audit
* **Interactive UI:** Users can review detected outliers in a dedicated "Audit Box."
* **Restore Capability:** If a valid data point (e.g., a "Non-Binary" gender or a "Young Genius") is flagged as an outlier, the user can check a box and **Restore** it to the clean dataset with a single click.

### 5. The "Showdown" Benchmark
* **Validation:** Automatically trains two models to prove the pipeline's value:
    1.  **Baseline:** Standard Mean Imputation + Linear Regression.
    2.  **Advanced:** The Self-Healing Pipeline + Random Forest Regressor.
* **Metrics:** Displays side-by-side **Mean Absolute Error (MAE)** and **R² Score** improvements.

## 🧰 Tech Stack & Models

### Core Infrastructure
* **Python 3.9+**: The backbone of the application.
* **Streamlit**: Used for the interactive frontend, real-time data editing, and visualization.
* **Pandas & NumPy**: High-performance data manipulation and vectorization.

### Machine Learning Engine (Scikit-Learn)
The pipeline utilizes a specific ensemble of models to achieve "Self-Healing":

| Component | Algorithm | Role |
| :--- | :--- | :--- |
| **Imputation** | `IterativeImputer` with `RandomForestRegressor` | Predicts missing values by modeling them as a function of other features (e.g., predicting *Salary* based on *Experience* and *Dept*). |
| **Outlier Detection** | `IsolationForest` | An unsupervised algorithm that isolates anomalies by randomly partitioning the feature space. Scores are used to flag "impossible" rows. |
| **Encoding** | `OrdinalEncoder` | Safely transforms categorical text (e.g., "HR", "Sales") into integer arrays for ML processing and decodes them back for the user. |
| **Benchmarking** | `LinearRegression` vs `RandomForestRegressor` | The "Showdown" module trains these two models on the fly to calculate the performance gap (MAE/R²) between the raw and repaired data. |

## 🛠️ Architecture

The pipeline follows a strict `Encode -> Impute -> Filter -> Decode` logic to ensure mathematical stability:

1.  **Ingestion:** Raw CSV is uploaded via Streamlit.
2.  **Encoding:** * Text columns are mapped to integers (e.g., HR=0, Sales=1) using `OrdinalEncoder`.
    * Encoders are stored in memory to allow reverse-translation later.
3.  **Imputation (The Builder):** * `IterativeImputer` cycles through features, using a **Random Forest** to model missing data as a function of other features.
4.  **Outlier Detection (The Bouncer):** * `IsolationForest` assigns an "Anomaly Score" to every row.
    * The worst 5% (configurable) are flagged.
5.  **Reasoning Extraction:** * The system analyzes the mathematical contribution of each feature to the Anomaly Score to generate human-readable explanations.
6.  **Decoding:** * Cleaned integer data is mapped back to original text labels.
7.  **Delivery:** * User downloads the clean CSV or restores outliers manually.

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* Pip

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ml-healing-pipeline.git](https://github.com/your-username/ml-healing-pipeline.git)
cd ml-healing-pipeline
```
### 2. Install Dependencies
```Bash
pip install -r requirements.txt
```

### 3. Generate Test Data (Optional)
Create a challenging synthetic dataset with missing categories, context mismatches, and outliers:
```Bash
python generate_messy_data.py
```

### 4. Run the Application
```Bash
streamlit run app.py
```

## 📂 Project Structure
* `app.py`: The frontend interface (Streamlit). Handles UI state, file uploads, and the interactive Audit/Restore logic.
* `processor.py`: The logic core. Contains the MachineLearningRepairKit class which manages encoding, imputation, and outlier detection.
* `generate_messy_data.py`: A utility script to generate synthetic datasets for stress-testing the pipeline.

## 🔮 Future Roadmap

* **Classification Support:** Enable the "Showdown" to evaluate categorical targets (Accuracy/F1-Score).
* **Deep Learning Imputer:** Experiment with Autoencoders for complex non-linear imputation.
* **API Mode:** Expose the processor.py logic via FastAPI for headless integration.
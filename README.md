# Customer Churn Prediction

A machine-learning project that predicts whether a telecom customer is likely to leave (churn) based on selected customer attributes. The project includes a data-analysis and model-training notebook, saved model/scaler files, a CSV dataset, and a Streamlit interface intended for interactive predictions.

> **Project status:** The notebook and model artifacts are included. The Streamlit app currently needs a few code/path corrections before it can run successfully (details in the Troubleshooting section).


## 1. Project Objective

Customer churn means a customer stops using a company's service. Predicting churn can help a telecom business identify customers who may leave and prioritize retention actions.

This project uses customer information to classify churn into:
- **Yes (1):** the customer is predicted to churn.
- **No (0):** the customer is predicted to stay.

The prediction is a classification result, not a guarantee of customer behavior.

## 2. Project Structure

```text
Custmer Churn Prediction/
├── app.py                     # Streamlit prediction interface
├── customer_churn_data.csv     # Dataset used for analysis and training
├── Untitled9.ipynb            # Exploratory analysis, preprocessing, training, evaluation
├── best_model.pkl              # Saved best Random Forest model from GridSearchCV
└── scaler.pkl                  # Saved StandardScaler
```

## 3. Dataset

The dataset file is `customer_churn_data.csv`. Its columns are:

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Age` | Customer age |
| `Gender` | Customer gender (`Male` / `Female`) |
| `Tenure` | Length of time the customer has been with the service, represented in the dataset as a numeric value |
| `MonthlyCharges` | Customer's monthly service charge |
| `ContractType` | Contract category (e.g., month-to-month, one-year, two-year) |
| `InternetService` | Internet service type; missing values are filled with `No` in the notebook |
| `TotalCharges` | Total charges recorded for the customer |
| `TechSupport` | Whether the customer has technical support (`Yes` / `No`) |
| `Churn` | Target label (`Yes` / `No`) |

The notebook's model is trained using only these four input features: **Age, Gender, Tenure, and MonthlyCharges**. The target is `Churn`. The other dataset columns are used for inspection/analysis but are not included in the model input selection shown in the notebook.

## 4. Notebook Workflow

The notebook (`Untitled9.ipynb`) follows these main steps:

1. **Load libraries and data** — imports pandas, NumPy, Matplotlib, Seaborn, and loads the CSV dataset.
2. **Initial inspection** — views sample rows, dataset tail, column types, missing-value counts, duplicate count, and descriptive statistics.
3. **Missing-value handling** — fills missing values in `InternetService` with `No`.
4. **Exploratory analysis** — checks InternetService frequencies, numeric correlations, churn counts, and average MonthlyCharges, Tenure, and Age by churn; also examines monthly charges by contract type and gender/churn group.
5. **Feature and target selection** — selects `Age`, `Gender`, `Tenure`, `MonthlyCharges` as features and `Churn` as target.
6. **Encoding** — LabelEncoder transforms the target and Gender into numeric values. The notebook comments intend Churn `Yes = 1`, `No = 0`, and Gender `Male = 1`, `Female = 0`.
7. **Train/test split** — uses 80% training data and 20% test data with `random_state=42`.
8. **Scaling** — applies `StandardScaler` to feature values and saves the scaler as `scaler.pkl`.
9. **Model training and evaluation** — trains several classifiers and evaluates predictions using Accuracy Score.
10. **Hyperparameter tuning** — uses GridSearchCV with 5-fold cross-validation for SVC, Decision Tree, and Random Forest.
11. **Save final estimator** — saves the best Random Forest estimator selected by GridSearchCV to `best_model.pkl`.

## 5. Machine-Learning Models

The notebook trains and evaluates these classification approaches:


### Logistic Regression
A linear classification model used as an initial baseline for churn prediction.

### K-Nearest Neighbors (KNN)
Predicts a class using the labels of nearby samples in feature space.

### Support Vector Classifier (SVC)
Uses a support-vector decision boundary. GridSearchCV searches the values of `C` and `kernel` listed in the notebook:
- `C`: `0.01`, `0.1`, `0.5`, `1`
- `kernel`: `linear`, `poly`, `rbf`, `sigmoid`

### Decision Tree Classifier
Learns decision rules by splitting feature values. The notebook searches criterion, splitter, maximum depth, minimum split size, and minimum leaf size using GridSearchCV.

### Random Forest Classifier
Combines multiple decision trees. GridSearchCV searches:
- `n_estimators`: 32, 64, 128, 256
- `max_features`: 2, 3, 4
- `bootstrap`: True / False

The notebook selects `grid_rfc.best_estimator_` as `best_model` and saves it. **The notebook code does not document numerical scores in this README; run the notebook to see the actual results in your environment.**

## 6. Evaluation

The notebook defines a `modelperformance()` helper that reports **Accuracy Score** by comparing predictions with the test labels. Accuracy is the fraction of test samples classified correctly.

For a churn use case, accuracy alone may not be enough—especially if the churn/non-churn classes are imbalanced. Consider adding precision, recall, F1-score, a confusion matrix, and ROC-AUC to assess how well the model identifies customers who churn.

## 7. Saved Files and Their Purpose

- `best_model.pkl`: the tuned Random Forest estimator selected by GridSearchCV.
- `scaler.pkl`: the fitted StandardScaler used to scale the model's input features.

These files are created by the notebook. Keep the model and scaler consistent with the feature order used during training: **Age, Gender, Tenure, MonthlyCharges**.

## 8. Streamlit Application

`app.py` is intended to provide a simple interactive UI. The user enters:
- Age
- Tenure
- Monthly Charges
- Gender

When Predict is clicked, the app encodes Gender (`Male` as 1, `Female` as 0), scales the input with the saved scaler, calls the model's `predict()` method, and displays `Yes` or `No`.

### Intended prediction flow

```text
User inputs
    ↓
Encode Gender
    ↓
Arrange features: Age, Gender, Tenure, MonthlyCharges
    ↓
Scale with scaler.pkl
    ↓
Predict with saved model
    ↓
Display Churn: Yes / No
```

## 9. Installation

Use Python 3.9+ (a compatible Python environment is recommended).

1. Open a terminal in the project directory.
2. Create and activate a virtual environment if desired:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```
   **macOS/Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 10. Run the Notebook

Open the notebook in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open `Untitled9.ipynb` and run the cells in order. The notebook's first data-download cell uses the Kaggle CLI; if you already have `customer_churn_data.csv` in the project directory, you can skip the download cell and use the local CSV. Kaggle download requires Kaggle CLI setup and credentials.

## 11. Run the Streamlit App

After correcting the app issues listed below, launch it with:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## 12. Known Issues to Fix in `app.py`

The provided `app.py` has these problems:

1. **Import typo:** `imort joblib` should be `import joblib`.
2. **Model filename mismatch:** the app loads `model.pkl`, but the project contains `best_model.pkl`. Update the model-loading line to use the actual filename, or rename the model file consistently.
3. **Input feature order:** verify the app's input order is identical to the model training order: `Age`, `Gender`, `Tenure`, `MonthlyCharges`.
4. **Scaling consistency:** the notebook currently calls `scaler.fit_transform()` on both training and test sets. The correct approach is to fit the scaler on training data only and call `scaler.transform()` on test data. The app should use the scaler fitted on the training data.
5. **Model/scaler consistency:** the scaler and model files should be regenerated together after correcting the notebook preprocessing, then used by the app.

Until these are corrected and tested, the app may fail to start or may produce unreliable predictions.

## 13. Suggested Improvements

- Fix the notebook's test scaling to use `transform()` rather than fitting a new scaler on test data.
- Make the train/test split stratified by `Churn` (`stratify=y`) to preserve class distribution.
- Evaluate with Precision, Recall, F1-score, confusion matrix, and ROC-AUC, in addition to accuracy.
- Check class imbalance and consider class weights or resampling if justified by analysis.
- Save preprocessing and model together in a scikit-learn Pipeline to reduce mismatch risk.
- Add input validation and clear explanation of prediction results to the Streamlit UI.
- Use a descriptive notebook filename instead of `Untitled9.ipynb`.

## 14. Requirements

Install project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The requirements file should include the packages used by the notebook and app (such as pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit, and Jupyter).

## 15. Disclaimer

This project is for educational and analytical purposes. A churn prediction should be interpreted as a model estimate, not a certainty. Business decisions should consider additional customer context and model evaluation results.

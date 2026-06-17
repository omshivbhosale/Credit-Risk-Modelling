# Credit Risk Modelling

A machine learning web application that predicts the **probability of loan default** for an applicant and converts it into a **credit score (300–900)** with an interpretable risk rating (Poor / Average / Good / Excellent). Built with a Logistic Regression model and a Streamlit front end.

> This project was built as a hands-on implementation while following the **codebasics ML course**. I used it to learn end-to-end ML deployment: feature engineering, model persistence, score calibration, and building an interactive prediction UI.

---

## What it does

The user enters applicant and loan details (age, income, loan amount, tenure, delinquency history, credit utilization, residence type, loan purpose, loan type). The app then:

1. Engineers features (e.g. loan-to-income ratio) and one-hot encodes categorical inputs.
2. Scales the inputs using a fitted `MinMaxScaler`.
3. Runs a trained **Logistic Regression** model to estimate default probability.
4. Converts that probability into a **300–900 credit score** and a rating band.

## Tech stack

- **Python**
- **scikit-learn** — Logistic Regression, MinMaxScaler
- **pandas / numpy** — data handling and scoring math
- **Streamlit** — interactive web UI
- **joblib** — model serialization

## Model details

- **Algorithm:** Logistic Regression (binary classification: default vs. non-default)
- **Features used (13):** age, loan tenure, number of open accounts, credit utilization ratio, loan-to-income ratio, delinquency ratio, average DPD per delinquency, plus one-hot encoded residence type, loan purpose, and loan type
- **Score conversion:** the model's non-default probability is mapped onto a 300–900 scale (`base_score = 300`, `scale = 600`), then bucketed into Poor (300–499), Average (500–649), Good (650–749), and Excellent (750–900)

## Project structure

```
Credit-Risk-Modelling/
├── main.py                 # Streamlit app (UI + input handling)
├── prediction_helper.py    # Preprocessing, scaling, scoring logic
├── artifacts/
│   └── model_data.joblib   # Trained model, scaler, feature list
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup and run

1. **Clone the repository**
   ```bash
   git clone https://github.com/omshivbhosale/Credit-Risk-Modelling.git
   cd Credit-Risk-Modelling
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run main.py
   ```
   The app opens in your browser at `http://localhost:8501`.

## What I learned

- Building an end-to-end ML pipeline from raw inputs to a deployed prediction UI
- Translating a raw model probability into a business-friendly credit score and rating
- Feature engineering (loan-to-income ratio) and consistent train/inference preprocessing
- Serializing and loading models with joblib for reuse in a separate inference script

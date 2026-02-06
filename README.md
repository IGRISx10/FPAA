# ⚽ Football Player Injury Category Prediction

This project implements a **machine learning–based decision-support system** to analyze and predict **football injury categories** using player and seasonal context.

The system is developed strictly for **academic and analytical purposes** and follows a structured, multi-phase machine learning pipeline aligned with existing research in sports injury analytics.

---

## 📌 Project Objective

To predict the **category of injury** (e.g., Soft Tissue Injury, Severe Knee Injury) using:
- Player age
- Playing position
- Seasonal context

The model does **not** predict injury occurrence or provide medical diagnosis.  
It demonstrates how machine learning models can **learn injury patterns** from historical data.

---

## 🧠 Methodology Overview

The project was developed in **7 structured phases**:

1. **Data Understanding & Alignment**
2. **Data Cleaning & Target Definition**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Engineering & Encoding**
5. **Baseline Model Training**
6. **Target Refinement & Model Improvement**
7. **Final Ensemble Model & Evaluation**

A **Voting Ensemble Classifier** combining:
- Logistic Regression
- Random Forest
- Gradient Boosting  

was used as the final model.

---

## 🏥 Injury Categories Predicted

The final target variable (`INJURY_CATEGORY`) includes:

- Soft Tissue Injury  
- Severe Knee Injury  
- Lower Limb Injury  
- Upper Body Injury  
- Minor / Other  

This grouping reduces class imbalance and reflects **clinically meaningful injury types**.

---

## 📊 Model Performance (Final)

- **Top-1 Accuracy:** ~34%
- **Weighted F1-Score:** ~0.33
- **Top-2 Accuracy:** ~61%

> Top-2 accuracy indicates that in over 60% of cases, the correct injury category appears within the model’s top two predictions — a realistic metric for medical decision-support systems.

---

## 🖥️ Streamlit User Interface

A Streamlit-based UI is provided to:
- Accept player context inputs
- Display **Top-1 and Top-2 predictions**
- Visualize model confidence scores
- Demonstrate how the model behaves interactively

The UI is intended **only as a prototype demonstration**.

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
python -m pip install -r requirements.txt

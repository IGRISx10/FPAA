import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =========================================
# 1. LOAD & PREPARE DATA
# =========================================
try:
    df = pd.read_csv("Training_Phase6_Master.csv")
except FileNotFoundError:
    df = pd.read_csv("Training_Phase4_Cleaned.csv")

    category_mapping = {
        'Severe Knee Injury': 'Severe Knee Injury',
        'ACL injury': 'Severe Knee Injury',
        'Cruciate ligament injury': 'Severe Knee Injury',
        'Muscle Injury': 'Soft Tissue Injury',
        'Hamstring Injury': 'Soft Tissue Injury',
        'Thigh Injury': 'Soft Tissue Injury',
        'Groin Injury': 'Soft Tissue Injury',
        'Calf Injury': 'Soft Tissue Injury',
        'Ankle Injury': 'Lower Limb Injury',
        'Foot Injury': 'Lower Limb Injury',
        'Achilles Injury': 'Lower Limb Injury',
        'Shoulder injury': 'Upper Body Injury',
        'Back injury': 'Upper Body Injury'
    }

    df['INJURY_CATEGORY'] = df['INJURY_TYPE'].map(category_mapping).fillna('Minor/Other')

print("✅ Data Loaded. Target Distribution:")
print(df['INJURY_CATEGORY'].value_counts())

# =========================================
# 2. ENCODING PIPELINE
# =========================================
X = pd.get_dummies(
    df[['POSITION', 'SEASON', 'AGE']],
    columns=['POSITION', 'SEASON']
)

y = df['INJURY_CATEGORY']

# Safety check
if not X.select_dtypes(include=['object']).empty:
    raise ValueError("❌ Error: Text columns still present in X!")

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# 3. DEFINE ENSEMBLE MODELS
# =========================================
print("\n⚙️ Initializing Ensemble Models...")

clf1 = LogisticRegression(
    class_weight='balanced',
    max_iter=3000,
    random_state=42
)

clf2 = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    random_state=42
)

clf3 = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

# FINAL ENSEMBLE MODEL
eclf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('gb', clf3)
    ],
    voting='soft'
)

# =========================================
# 4. TRAINING
# =========================================
print("🚀 Training Ensemble Model...")
eclf.fit(X_train, y_train)
print("✅ Training Complete.")

# =========================================
# 5. STANDARD EVALUATION
# =========================================
y_pred = eclf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🏆 Ensemble Top-1 Accuracy: {acc:.2%}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# =========================================
# 6. TOP-2 ACCURACY (IMPORTANT)
# =========================================
probs = eclf.predict_proba(X_test)

top2_preds = np.argsort(probs, axis=1)[:, -2:]
class_labels = list(eclf.classes_)

y_test_idx = y_test.map({label: i for i, label in enumerate(class_labels)})
top2_acc = np.mean([true in pred for true, pred in zip(y_test_idx, top2_preds)])

print(f"🎯 Top-2 Accuracy (Medical Relevance): {top2_acc:.2%}")

# =========================================
# 7. CONFUSION MATRIX
# =========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(
    f"Ensemble Confusion Matrix\n"
    f"Top-1 Acc: {acc:.1%} | Top-2 Acc: {top2_acc:.1%}"
)
plt.show()

# =========================================
# 8. SAVE MODEL & FEATURE SCHEMA (FOR UI)
# =========================================
joblib.dump(eclf, "ensemble_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("✅ Model and feature schema saved successfully.")

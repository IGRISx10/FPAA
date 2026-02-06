import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD DATA
df = pd.read_csv("Training_Phase6_Master.csv")

# 2. ENCODING (The Fix)
# This converts 'Defensive Midfield' -> 'POSITION_Defensive Midfield' (1 or 0)
# It automatically drops the original text columns.
X = pd.get_dummies(df[['POSITION', 'SEASON', 'AGE']], columns=['POSITION', 'SEASON'])
y = df['INJURY_CATEGORY']

# 3. VERIFY (Safety Check)
print("Text columns remaining in X:", X.select_dtypes(include=['object']).columns.tolist())
# Output must be: []

# 4. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. RETRAIN MODELS
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. RESULTS
print("\n--- FINAL CLASSIFICATION REPORT ---")
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))


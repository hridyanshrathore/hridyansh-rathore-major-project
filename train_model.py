# ----------------------------------------------------------
# train_model.py  (works with dataset.csv from your ZIP)
# ----------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------- 1) Load dataset ----------
# Use latin1 to avoid occasional encoding issues on Windows
df = pd.read_csv("dataset.csv", encoding="latin1")

# Normalize column names (strip spaces / keep originals for safety)
df.columns = [c.strip() for c in df.columns]

# Target column in this dataset is usually 'Disease'
if "Disease" not in df.columns:
    raise ValueError("Could not find 'Disease' column in dataset.csv")

# Symptom columns are named like Symptom_1 ... Symptom_17 (case-insensitive)
symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
if not symptom_cols:
    raise ValueError("No symptom columns found (expected Symptom_1 ...).")

# Clean symptom strings
for c in symptom_cols:
    df[c] = (
        df[c]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

# ---------- 2) Build one-hot matrix of all symptoms ----------
# Collect unique symptom names
sym_set = set()
for c in symptom_cols:
    sym_set.update(df[c].unique().tolist())
sym_set.discard("")  # remove blanks

symptoms = sorted(sym_set)

# Create binary matrix
X = pd.DataFrame(0, index=df.index, columns=symptoms, dtype=np.uint8)
for i, row in df[symptom_cols].iterrows():
    for s in row.values:
        if s:
            X.at[i, s] = 1

y = df["Disease"].astype(str)

# ---------- 3) Encode labels ----------
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ---------- 4) Train/validate ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
acc = rf.score(X_val, y_val)
print(f"Model trained successfully. Validation accuracy: {acc*100:.2f}%")


# ---------- 5) Save artifacts ----------
joblib.dump(rf, "disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(symptoms, "symptom_columns.pkl")  # needed for the app

print("Model and encoder files saved successfully.")


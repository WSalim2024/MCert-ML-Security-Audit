import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# ============================================================
# VULNERABLE ML PIPELINE - FOR DEMONSTRATION ONLY
# ============================================================
# This code intentionally contains security vulnerabilities
# to demonstrate common ML security risks.
# DO NOT USE IN PRODUCTION!
# ============================================================

print("=" * 60)
print("  VULNERABLE ML PIPELINE - INSECURE IMPLEMENTATION")
print("=" * 60)
print()

# --- VULNERABILITY 1: No Data Validation/Sanitization ---
# Risk: CSV injection, malformed data, DoS attacks
print("STEP 1: Loading data (NO VALIDATION)")
print("-" * 40)
data = pd.read_csv('user_data.csv')
print(f"Loaded data: {len(data)} rows (no validation performed)")
print()

# --- VULNERABILITY 2: No Input Validation ---
# Risk: Assuming column structure never changes
print("STEP 2: Splitting features (NO INPUT VALIDATION)")
print("-" * 40)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(f"Features: {X.shape[1]} columns")
print(f"Target classes: {y.nunique()}")
print()

# --- VULNERABILITY 3: Fixed/Predictable Random State ---
# Risk: Attackers can predict data splits and exploit model behavior
print("STEP 3: Train/Test Split (FIXED RANDOM STATE)")
print("-" * 40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42  # VULNERABLE: Predictable!
)
print(f"[WARNING] Using fixed random_state=42 (PREDICTABLE)")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print()

# --- MODEL TRAINING ---
print("STEP 4: Training Model")
print("-" * 40)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model trained. Accuracy: {accuracy:.2%}")
print()

# --- VULNERABILITY 4: Insecure Serialization with Pickle ---
# Risk: Pickle allows arbitrary code execution when loading untrusted files
print("STEP 5: Saving Model (INSECURE PICKLE)")
print("-" * 40)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"[WARNING] Model saved as 'model.pkl' (UNENCRYPTED)")
print(f"[WARNING] Using pickle (VULNERABLE TO RCE)")
print()

# --- VULNERABILITY 5: No Integrity Checks ---
# Risk: Model can be tampered with without detection
print("STEP 6: No Integrity Verification")
print("-" * 40)
print("[WARNING] No hash generated for model file")
print("[WARNING] No way to detect tampering")
print()

print("=" * 60)
print("  VULNERABLE PIPELINE COMPLETE")
print("=" * 60)
print()
print("Security Issues:")
print("  [X] No input validation or sanitization")
print("  [X] Predictable random state (42)")
print("  [X] Unencrypted model storage")
print("  [X] Pickle serialization (RCE risk)")
print("  [X] No integrity verification")

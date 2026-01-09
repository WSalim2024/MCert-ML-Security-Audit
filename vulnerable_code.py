import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# --- VULNERABILITY DEMO ---

# 1. Load dataset (Flaw: No validation/sanitization)
# Risk: CSV injection or malformed data crashing the system
data = pd.read_csv('user_data.csv')

# 2. Split features (Flaw: No input validation)
# Risk: Assuming column structure never changes
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3. Train Test Split (Flaw: Fixed/Predictable Random State)
# Risk: Attackers can predict data splits if seed is static/known
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Serialization (Flaw: Using Pickle)
# Risk: Pickle is insecure; loading untrusted pickles allows Remote Code Execution (RCE)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
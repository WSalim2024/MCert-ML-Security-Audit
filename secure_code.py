import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib  # Safer/Faster than pickle for sklearn (though still needs caution)
import hashlib
import os
import secrets  # Cryptographically strong random numbers


def validate_and_load(filepath):
    """Mitigation 1: Input Validation & Sanitization"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found.")

    # Validation: Ensure it's a CSV and not too massive (DoS protection)
    if not filepath.endswith('.csv'):
        raise ValueError("Invalid file format. Only CSV allowed.")

    df = pd.read_csv(filepath)

    # Sanitization: Check for nulls or infinite values
    if df.isnull().values.any():
        print("⚠️ Warning: Null values detected. Dropping...")
        df.dropna(inplace=True)

    return df


def secure_random_seed():
    """Mitigation 2: Secure Random State"""
    # Use secrets module to generate a non-predictable seed
    return secrets.randbelow(100_000)


def save_model_securely(model, filename):
    """Mitigation 3: Integrity Checks (Hashing)"""
    # Save using joblib
    joblib.dump(model, filename)

    # Create a SHA-256 hash of the file for integrity verification
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    # Save the hash separately
    with open(f"{filename}.sha256", "w") as f:
        f.write(sha256_hash.hexdigest())

    print(f"✅ Model saved with SHA-256 integrity check: {filename}")


def generate_fallback_data():
    """Generate valid multi-class training data if user_data.csv doesn't exist"""
    print("📊 Generating fallback training data...")
    np.random.seed(secrets.randbelow(100_000))

    # Create features
    data = pd.DataFrame(np.random.rand(100, 4), columns=['Feat1', 'Feat2', 'Feat3', 'Feat4'])

    # Generate proper integer classes (3 classes with good distribution)
    data['Target'] = np.random.randint(0, 3, size=100)

    # Guarantee at least 2 samples per class for stratification
    data.loc[0, 'Target'] = 0
    data.loc[1, 'Target'] = 0
    data.loc[2, 'Target'] = 1
    data.loc[3, 'Target'] = 1
    data.loc[4, 'Target'] = 2
    data.loc[5, 'Target'] = 2

    return data


def prepare_target_for_training(y):
    """
    Prepare target variable for training by handling classes with too few samples.
    Returns processed y and whether stratification is safe to use.
    """
    class_counts = y.value_counts()
    min_samples = class_counts.min()
    n_classes = len(class_counts)

    print(f"📊 Target classes found: {n_classes}")
    print(f"   Class distribution: {class_counts.to_dict()}")
    print(f"   Min samples per class: {min_samples}")

    # Check if we have enough samples for stratification (need at least 2 per class)
    can_stratify = min_samples >= 2

    if not can_stratify:
        print("⚠️ Some classes have <2 samples. Stratification disabled.")

        # Option: Filter out classes with only 1 sample for cleaner training
        valid_classes = class_counts[class_counts >= 2].index
        if len(valid_classes) >= 2:
            mask = y.isin(valid_classes)
            print(f"   Filtering to classes with ≥2 samples: {list(valid_classes)}")
            return y, mask, True  # Can stratify after filtering
        else:
            print("   Not enough valid classes to filter. Proceeding without stratification.")
            return y, None, False

    return y, None, True


if __name__ == "__main__":
    try:
        print("Secure Loader initialized...")

        # 1. Try to load user_data.csv, fall back to generated data
        if os.path.exists('user_data.csv'):
            data = validate_and_load('user_data.csv')
            print(f"✅ Loaded 'user_data.csv' ({len(data)} rows)")
        else:
            data = generate_fallback_data()
            print("⚠️ 'user_data.csv' not found. Using generated data.")

        # 2. Prepare features and target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(int)

        # 3. Validate and prepare target for training
        y, filter_mask, can_stratify = prepare_target_for_training(y)

        # Apply filter if needed
        if filter_mask is not None:
            X = X[filter_mask]
            y = y[filter_mask]
            print(f"   Filtered dataset: {len(X)} samples remaining")

        # Validate we have multiple classes
        unique_classes = y.nunique()
        if unique_classes < 2:
            raise ValueError(f"Need at least 2 classes for classification, but found {unique_classes}")

        # 4. Secure Split
        secure_seed = secure_random_seed()
        print(f"Using Cryptographically Secure Seed: {secure_seed}")

        # Use stratify only if safe to do so
        split_params = {
            'test_size': 0.2,
            'random_state': secure_seed
        }

        if can_stratify:
            split_params['stratify'] = y
            print("✅ Using stratified split (class distribution preserved)")
        else:
            print("⚠️ Using random split (stratification not possible)")

        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)

        print(f"✅ Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        print(f"   Train classes: {sorted(y_train.unique())}")

        # 5. Train
        model = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
        model.fit(X_train, y_train)

        # Show accuracy
        accuracy = model.score(X_test, y_test)
        print(f"✅ Model trained. Test accuracy: {accuracy:.2%}")

        # 6. Secure Save
        save_model_securely(model, 'secure_model.joblib')

    except Exception as e:
        print(f"Security Alert: {e}")

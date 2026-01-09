import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import hashlib
import os
import secrets
from cryptography.fernet import Fernet


# ============================================================
# SECURITY MITIGATION FUNCTIONS
# ============================================================

def validate_and_load(filepath):
    """Mitigation 1: Input Validation & Sanitization"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found.")

    # Validation: Ensure it's a CSV and not too massive (DoS protection)
    if not filepath.endswith('.csv'):
        raise ValueError("Invalid file format. Only CSV allowed.")

    # Check file size (prevent DoS with massive files - limit 100MB)
    file_size = os.path.getsize(filepath)
    if file_size > 100 * 1024 * 1024:
        raise ValueError("File too large. Maximum size is 100MB.")

    df = pd.read_csv(filepath)

    # Sanitization: Check for nulls or infinite values
    if df.isnull().values.any():
        print("[WARNING] Null values detected. Dropping...")
        df.dropna(inplace=True)

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[numeric_cols].values).any():
        print("[WARNING] Infinite values detected. Replacing with NaN and dropping...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    return df


def secure_random_seed():
    """Mitigation 2: Secure Random State"""
    # Use secrets module to generate a non-predictable seed
    return secrets.randbelow(100_000)


def generate_or_load_key(key_path='encryption.key'):
    """
    Mitigation 4a: Secure Key Management
    Generate a new encryption key or load existing one.
    In production, this should be stored in a secure key vault (AWS KMS, Azure Key Vault, etc.)
    """
    if os.path.exists(key_path):
        print(f"[KEY] Loading existing encryption key from '{key_path}'")
        with open(key_path, 'rb') as f:
            key = f.read()
    else:
        print(f"[KEY] Generating new encryption key...")
        key = Fernet.generate_key()
        with open(key_path, 'wb') as f:
            f.write(key)
        print(f"[OK] Encryption key saved to '{key_path}'")
        print("[WARNING] In production, store this key in a secure key vault!")

    return key


def save_model_encrypted(model, filename, key):
    """
    Mitigation 4: Encrypted Model Saving
    Encrypts the model using Fernet symmetric encryption before saving.
    Also creates SHA-256 hash for integrity verification.
    """
    # Step 1: Serialize model to bytes using joblib
    import io
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()

    # Step 2: Encrypt the serialized model
    cipher = Fernet(key)
    encrypted_model = cipher.encrypt(model_bytes)

    # Step 3: Save encrypted model
    encrypted_filename = filename.replace('.joblib', '.encrypted')
    with open(encrypted_filename, 'wb') as f:
        f.write(encrypted_model)

    # Step 4: Create SHA-256 hash of the ENCRYPTED file for integrity verification
    sha256_hash = hashlib.sha256(encrypted_model).hexdigest()

    # Save the hash separately
    hash_filename = f"{encrypted_filename}.sha256"
    with open(hash_filename, 'w') as f:
        f.write(sha256_hash)

    print(f"[ENCRYPTED] Model encrypted and saved: {encrypted_filename}")
    print(f"[HASH] SHA-256 hash saved: {hash_filename}")
    print(f"   Hash: {sha256_hash[:32]}...")

    return encrypted_filename, hash_filename, sha256_hash


def load_model_encrypted(filename, key):
    """
    Mitigation 5: Load Encrypted Model with Integrity Check
    Decrypts the model and verifies integrity using SHA-256 hash.
    """
    encrypted_filename = filename.replace('.joblib', '.encrypted')
    hash_filename = f"{encrypted_filename}.sha256"

    # Step 1: Load encrypted model
    if not os.path.exists(encrypted_filename):
        raise FileNotFoundError(f"Encrypted model '{encrypted_filename}' not found.")

    with open(encrypted_filename, 'rb') as f:
        encrypted_model = f.read()

    # Step 2: Verify integrity (compare hash)
    if os.path.exists(hash_filename):
        with open(hash_filename, 'r') as f:
            expected_hash = f.read().strip()

        actual_hash = hashlib.sha256(encrypted_model).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                "[FAIL] INTEGRITY CHECK FAILED!\n"
                f"   Expected: {expected_hash[:32]}...\n"
                f"   Got:      {actual_hash[:32]}...\n"
                "   The model file may have been tampered with!"
            )
        print(f"[OK] Integrity check passed. Hash: {actual_hash[:32]}...")
    else:
        print("[WARNING] No hash file found. Skipping integrity check.")

    # Step 3: Decrypt model
    cipher = Fernet(key)
    try:
        decrypted_bytes = cipher.decrypt(encrypted_model)
    except Exception as e:
        raise ValueError(f"[FAIL] DECRYPTION FAILED! Wrong key or corrupted file. Error: {e}")

    # Step 4: Deserialize model
    import io
    buffer = io.BytesIO(decrypted_bytes)
    model = joblib.load(buffer)

    print(f"[OK] Model successfully decrypted and loaded!")
    return model


def save_model_securely(model, filename):
    """
    Legacy function for backward compatibility (unencrypted saving).
    Mitigation 3: Integrity Checks (Hashing only, no encryption)
    """
    # Save using joblib
    joblib.dump(model, filename)

    # Create a SHA-256 hash of the file for integrity verification
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    # Save the hash separately
    with open(f"{filename}.sha256", "w") as f:
        f.write(sha256_hash.hexdigest())

    print(f"[OK] Model saved (unencrypted) with SHA-256 integrity check: {filename}")


def generate_fallback_data():
    """Generate valid multi-class training data if user_data.csv doesn't exist"""
    print("[DATA] Generating fallback training data...")
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

    print(f"[DATA] Target classes found: {n_classes}")
    print(f"   Class distribution: {dict(sorted(class_counts.items()))}")
    print(f"   Min samples per class: {min_samples}")

    # Check if we have enough samples for stratification (need at least 2 per class)
    can_stratify = min_samples >= 2

    if not can_stratify:
        print("[WARNING] Some classes have <2 samples. Stratification disabled.")

        # Option: Filter out classes with only 1 sample for cleaner training
        valid_classes = class_counts[class_counts >= 2].index
        if len(valid_classes) >= 2:
            mask = y.isin(valid_classes)
            print(f"   Filtering to classes with >=2 samples: {list(valid_classes)}")
            return y, mask, True
        else:
            print("   Not enough valid classes to filter. Proceeding without stratification.")
            return y, None, False

    return y, None, True


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("  SECURE ML PIPELINE - WITH ENCRYPTION")
        print("=" * 60)
        print()

        # 1. Try to load user_data.csv, fall back to generated data
        print("STEP 1: Data Loading & Validation")
        print("-" * 40)
        if os.path.exists('user_data.csv'):
            data = validate_and_load('user_data.csv')
            print(f"[OK] Loaded 'user_data.csv' ({len(data)} rows)")
        else:
            data = generate_fallback_data()
            print("[WARNING] 'user_data.csv' not found. Using generated data.")
        print()

        # 2. Prepare features and target
        print("STEP 2: Feature Preparation")
        print("-" * 40)
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
        print()

        # 4. Secure Split
        print("STEP 3: Secure Train/Test Split")
        print("-" * 40)
        secure_seed = secure_random_seed()
        print(f"[RANDOM] Cryptographically Secure Seed: {secure_seed}")

        split_params = {
            'test_size': 0.2,
            'random_state': secure_seed
        }

        if can_stratify:
            split_params['stratify'] = y
            print("[OK] Using stratified split (class distribution preserved)")
        else:
            print("[WARNING] Using random split (stratification not possible)")

        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
        print(f"   Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        print(f"   Train classes: {sorted(y_train.unique())}")
        print()

        # 5. Train Model
        print("STEP 4: Model Training")
        print("-" * 40)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"[OK] Model trained successfully!")
        print(f"   Test accuracy: {accuracy:.2%}")
        print()

        # 6. Generate/Load Encryption Key
        print("STEP 5: Encryption Key Management")
        print("-" * 40)
        encryption_key = generate_or_load_key('encryption.key')
        print()

        # 7. Save Model with Encryption
        print("STEP 6: Encrypted Model Saving")
        print("-" * 40)
        encrypted_file, hash_file, model_hash = save_model_encrypted(
            model,
            'secure_model.joblib',
            encryption_key
        )
        print()

        # 8. Verification Test - Load the model back
        print("STEP 7: Verification - Loading Encrypted Model")
        print("-" * 40)
        loaded_model = load_model_encrypted('secure_model.joblib', encryption_key)

        # Verify loaded model works
        loaded_accuracy = loaded_model.score(X_test, y_test)
        print(f"   Loaded model accuracy: {loaded_accuracy:.2%}")

        if abs(loaded_accuracy - accuracy) < 0.0001:
            print("[OK] Model verification passed! Encryption/decryption successful.")
        else:
            print("[WARNING] Model accuracy differs after reload.")
        print()

        print("=" * 60)
        print("  SECURE PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Generated Files:")
        print(f"   [ENCRYPTED] Model: {encrypted_file}")
        print(f"   [HASH] Integrity:  {hash_file}")
        print(f"   [KEY] Encryption:  encryption.key")

    except Exception as e:
        print(f"[ERROR] Security Alert: {e}")
        import traceback

        traceback.print_exc()

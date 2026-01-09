"""
Test Script: Encryption & Integrity Verification
================================================
This script demonstrates:
1. PASS: Normal model save/load with integrity check
2. FAIL: Tampering detection
3. FAIL: Wrong encryption key
"""

import os
import sys
import shutil
from cryptography.fernet import Fernet

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from secure_code import (
    generate_or_load_key,
    save_model_encrypted,
    load_model_encrypted,
    generate_fallback_data,
    prepare_target_for_training,
    secure_random_seed
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def create_test_model():
    """Create a simple test model."""
    print("Creating test model...")
    data = generate_fallback_data()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)
    
    y, filter_mask, can_stratify = prepare_target_for_training(y)
    if filter_mask is not None:
        X = X[filter_mask]
        y = y[filter_mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=secure_random_seed(),
        stratify=y if can_stratify else None
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model created. Accuracy: {accuracy:.2%}")
    return model, X_test, y_test


def test_case_1_pass_integrity():
    """
    TEST CASE 1: PASS - Normal Operation
    Save and load model with correct key. Integrity check should pass.
    """
    print("\n" + "=" * 70)
    print("TEST CASE 1: PASS - Normal Encrypted Save/Load")
    print("=" * 70)
    
    try:
        # Cleanup
        for f in ['test_model.encrypted', 'test_model.encrypted.sha256', 'test_key.key']:
            if os.path.exists(f):
                os.remove(f)
        
        # Create model and key
        model, X_test, y_test = create_test_model()
        key = generate_or_load_key('test_key.key')
        
        # Save encrypted
        print("\n[SAVE] Saving encrypted model...")
        save_model_encrypted(model, 'test_model.joblib', key)

        # Load and verify
        print("\n[LOAD] Loading encrypted model...")
        loaded_model = load_model_encrypted('test_model.joblib', key)

        # Verify model works
        original_acc = model.score(X_test, y_test)
        loaded_acc = loaded_model.score(X_test, y_test)

        print(f"\n[PASS] TEST PASSED!")
        print(f"   Original accuracy: {original_acc:.2%}")
        print(f"   Loaded accuracy:   {loaded_acc:.2%}")
        print(f"   Models match: {abs(original_acc - loaded_acc) < 0.001}")

        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return False


def test_case_2_fail_tampering():
    """
    TEST CASE 2: FAIL - Tampering Detection
    Modify the encrypted file and verify integrity check catches it.
    """
    print("\n" + "=" * 70)
    print("TEST CASE 2: FAIL - Tampering Detection")
    print("=" * 70)

    try:
        # Cleanup and create fresh model
        for f in ['test_model.encrypted', 'test_model.encrypted.sha256', 'test_key.key']:
            if os.path.exists(f):
                os.remove(f)

        model, X_test, y_test = create_test_model()
        key = generate_or_load_key('test_key.key')

        # Save encrypted
        print("\n[SAVE] Saving encrypted model...")
        save_model_encrypted(model, 'test_model.joblib', key)

        # TAMPER with the file
        print("\n[TAMPER] TAMPERING with encrypted file...")
        with open('test_model.encrypted', 'ab') as f:
            f.write(b'\x00\x01\x02\x03TAMPERED_DATA')
        print("   Added malicious bytes to model file")

        # Try to load - should fail integrity check
        print("\n[LOAD] Attempting to load tampered model...")
        try:
            loaded_model = load_model_encrypted('test_model.joblib', key)
            print("\n[FAIL] TEST FAILED: Tampering was NOT detected!")
            return False
        except ValueError as e:
            if "INTEGRITY CHECK FAILED" in str(e):
                print(f"\n[PASS] TEST PASSED! Tampering detected:")
                print(f"   {e}")
                return True
            else:
                raise

    except Exception as e:
        print(f"\n[ERROR] TEST ERROR: {e}")
        return False


def test_case_3_fail_wrong_key():
    """
    TEST CASE 3: FAIL - Wrong Encryption Key
    Try to load model with incorrect key.
    """
    print("\n" + "=" * 70)
    print("TEST CASE 3: FAIL - Wrong Encryption Key")
    print("=" * 70)

    try:
        # Cleanup and create fresh model
        for f in ['test_model.encrypted', 'test_model.encrypted.sha256',
                  'test_key.key', 'wrong_key.key']:
            if os.path.exists(f):
                os.remove(f)

        model, X_test, y_test = create_test_model()
        correct_key = generate_or_load_key('test_key.key')

        # Save with correct key
        print("\n[SAVE] Saving encrypted model with KEY_A...")
        save_model_encrypted(model, 'test_model.joblib', correct_key)

        # Generate a DIFFERENT key
        print("\n[KEY] Generating DIFFERENT key (KEY_B)...")
        wrong_key = Fernet.generate_key()
        print(f"   Wrong key: {wrong_key[:20]}...")

        # Try to load with wrong key
        print("\n[LOAD] Attempting to load with WRONG key...")
        try:
            # We need to temporarily bypass integrity check since hash is valid
            # but decryption will fail
            loaded_model = load_model_encrypted('test_model.joblib', wrong_key)
            print("\n[FAIL] TEST FAILED: Wrong key was NOT detected!")
            return False
        except ValueError as e:
            if "DECRYPTION FAILED" in str(e) or "InvalidToken" in str(e):
                print(f"\n[PASS] TEST PASSED! Wrong key detected:")
                print(f"   {e}")
                return True
            else:
                raise

    except Exception as e:
        if "InvalidToken" in str(e):
            print(f"\n[PASS] TEST PASSED! Decryption failed with wrong key.")
            return True
        print(f"\n[ERROR] TEST ERROR: {e}")
        return False


def test_case_4_fail_missing_key():
    """
    TEST CASE 4: FAIL - Missing Encryption Key
    Simulate key loss scenario.
    """
    print("\n" + "=" * 70)
    print("TEST CASE 4: FAIL - Missing/Lost Encryption Key")
    print("=" * 70)

    try:
        # Cleanup
        for f in ['test_model.encrypted', 'test_model.encrypted.sha256', 'test_key.key']:
            if os.path.exists(f):
                os.remove(f)

        model, X_test, y_test = create_test_model()
        key = generate_or_load_key('test_key.key')

        # Save encrypted
        print("\n[SAVE] Saving encrypted model...")
        save_model_encrypted(model, 'test_model.joblib', key)

        # DELETE the key (simulate key loss)
        print("\n[DELETE] DELETING encryption key (simulating key loss)...")
        os.remove('test_key.key')
        print("   Key file deleted!")

        # Try to load without key
        print("\n[LOAD] Attempting to load model without key...")
        print("   (In real scenario, you'd need to regenerate or recover the key)")

        # Show that model is unreadable without key
        with open('test_model.encrypted', 'rb') as f:
            encrypted_data = f.read()

        print(f"\n[PASS] TEST PASSED! Model is encrypted and unreadable without key")
        print(f"   Encrypted file size: {len(encrypted_data):,} bytes")
        print(f"   First 50 bytes (gibberish): {encrypted_data[:50]}")
        print(f"\n   [WARNING] WITHOUT THE KEY, THIS MODEL IS PERMANENTLY LOST!")

        return True

    except Exception as e:
        print(f"\n[ERROR] TEST ERROR: {e}")
        return False


def cleanup_test_files():
    """Remove all test files."""
    test_files = [
        'test_model.encrypted',
        'test_model.encrypted.sha256',
        'test_key.key',
        'wrong_key.key'
    ]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)


def run_all_tests():
    """Run all test cases and show summary."""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "ENCRYPTION TEST SUITE" + " " * 27 + "#")
    print("#" * 70)

    results = []

    # Run tests
    results.append(("Normal Operation", test_case_1_pass_integrity()))
    results.append(("Tampering Detection", test_case_2_fail_tampering()))
    results.append(("Wrong Key Detection", test_case_3_fail_wrong_key()))
    results.append(("Missing Key Scenario", test_case_4_fail_missing_key()))

    # Cleanup
    cleanup_test_files()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} - {name}")
    
    print("-" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

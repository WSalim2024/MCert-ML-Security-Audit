# 🛡️ ML Security Audit Report

**Project:** Machine Learning Code Security Audit  
**Date:** January 2026  
**Author:** Waqar Salim  
**Repository:** [MCert-ML-Security-Audit](https://github.com/WSalim2024/MCert-ML-Security-Audit)

---

## 🔍 Executive Summary
This audit analyzed a legacy Machine Learning training pipeline for common security vulnerabilities. The review identified three critical flaws related to **Insecure Deserialization**, **Input Validation**, and **Predictable Randomness**. 

A remediation plan was executed, resulting in a secured pipeline that utilizes cryptographic best practices and integrity verification. A Streamlit-based **Vulnerability Simulator** was developed to demonstrate the contrast between the flawed and secured implementations.

---

## 🚨 Vulnerability Analysis

### 1. Insecure Deserialization (Critical)
* **Vulnerability:** The code used the standard `pickle` library to serialize the trained model.
* **Risk:** `pickle` is inherently insecure. Malicious actors can construct a "poisoned" pickle file that executes arbitrary code (Remote Code Execution - RCE) on the server when loaded.
* **Status:** 🔴 **VULNERABLE** in `vulnerable_code.py`.

### 2. Lack of Input Validation (High)
* **Vulnerability:** The pipeline accepted any CSV input without checking file structure, size, or data distribution.
* **Risk:** Vulnerable to **Denial of Service (DoS)** via massive files, or runtime crashes due to malformed data (e.g., "One Class" errors in Logistic Regression).
* **Status:** 🔴 **VULNERABLE** in `vulnerable_code.py`.

### 3. Predictable Random State (Medium)
* **Vulnerability:** The model training used a hardcoded `random_state=42`.
* **Risk:** Attackers knowing the seed can predict data splits and craft "poisoned" inputs that specifically target the training set to bias the model (Model Poisoning).
* **Status:** 🔴 **VULNERABLE** in `vulnerable_code.py`.

---

## ✅ Remediation & Fixes

### 1. Secure Serialization & Integrity Checks
* **Fix:** Replaced `pickle` with `joblib` (safer for numerical arrays).
* **Enhancement:** Implemented **SHA-256 Hashing**. A separate `.sha256` checksum file is generated alongside the model. The loader verifies this hash before execution to prevent tampering.
* **Status:** 🟢 **FIXED** in `secure_code.py`.

### 2. Robust Input Sanitization
* **Fix:** Added a `validate_and_load()` function that:
    * Verifies file extension (`.csv`).
    * Checks for null/infinite values.
    * Ensures the target variable contains at least 3 classes to prevent mathematical solver crashes.
* **Status:** 🟢 **FIXED** in `secure_code.py` & `generate_test_data.py`.

### 3. Cryptographic Randomness
* **Fix:** Replaced hardcoded seeds with Python's `secrets` module (`secrets.randbelow()`) to generate non-deterministic, cryptographically strong seeds for data splitting.
* **Status:** 🟢 **FIXED** in `secure_code.py`.

---

## 🖥️ Vulnerability Simulator (Dashboard)
An interactive **Streamlit Dashboard** (`app.py`) was created

## ✍️ Author
#### Waqar Salim

## ⚠️ Disclaimer
This project is intended for educational purposes as part of a Master's degree portfolio. The code labeled "vulnerable" contains intentional security flaws for demonstration and must not be used in production environments. The "secure" implementations represent best practices for this specific context but should be integrated into a wider defense-in-depth strategy for enterprise deployment.
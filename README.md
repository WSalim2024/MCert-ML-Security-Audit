# 🛡️ ML Code Security Audit: Vulnerability Simulator

Interactive demonstration of ML security vulnerabilities and their mitigations, focusing on **Insecure Deserialization**, **Random State Flaws**, and **Unencrypted Model Storage**.

## 🎯 Features

- Side-by-side comparison of vulnerable vs. secure ML code
- **Fernet AES-128 encryption** for model storage
- **SHA-256 integrity verification** to detect tampering
- **Cryptographically secure random seeds** using Python's `secrets` module
- Interactive tampering simulation
- Comprehensive audit report

## 📋 Prerequisites

```bash
Python 3.8+
pip install streamlit pandas scikit-learn cryptography joblib
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/MCert-ML-Security-Audit.git
cd MCert-ML-Security-Audit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Test Data
```bash
python generate_test_data.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

### 5. Run Security Tests
```bash
python test_encryption.py
```

## 📁 Project Structure

```
MCert-ML-Security-Audit/
├── app.py                  # Streamlit UI application
├── secure_code.py          # Secure ML pipeline with encryption
├── vulnerable_code.py      # Vulnerable ML pipeline (demo only)
├── generate_test_data.py   # Test data generator
├── test_encryption.py      # Encryption test suite
├── AUDIT_REPORT.md         # Security audit findings
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🔐 Security Features Implemented

| Vulnerability | Mitigation | Implementation |
|--------------|------------|----------------|
| No data validation | Input sanitization | `validate_and_load()` |
| Fixed random state | Cryptographic RNG | `secrets.randbelow()` |
| Unencrypted storage | AES-128 encryption | `Fernet` encryption |
| No integrity check | SHA-256 hashing | Hash verification on load |
| Pickle RCE risk | Safer serialization | `joblib` with encryption |

## 🧪 Test Cases

Run `python test_encryption.py` to execute:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Normal Operation | Save/load encrypted model | ✅ Success |
| Tampering Detection | Modify encrypted file | ❌ Integrity check fails |
| Wrong Key | Use different encryption key | ❌ Decryption fails |
| Missing Key | Delete key file | ❌ Model unreadable |

### Main Interface
- **Left Panel**: Vulnerable implementation with security warnings
- **Right Panel**: Secure implementation with encryption status

### Integrity Verification
- Verify model integrity with one click
- Simulate tampering attacks to test detection

## ⚠️ Security Warnings

1. **Encryption Key**: In production, store `encryption.key` in a secure key vault (AWS KMS, Azure Key Vault)
2. **Never commit keys**: Add `*.key` to `.gitignore`
3. **Key rotation**: Implement 90-day key rotation policy

## 📝 License

For educational purposes only.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## ✍️ Author

**Waqar Salim**

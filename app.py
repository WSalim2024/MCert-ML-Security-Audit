import streamlit as st
import os
import sys
import subprocess
import hashlib

# --- Page Config ---
st.set_page_config(page_title="ML Security Audit", page_icon="🔐", layout="wide")

# --- Initialize Session State ---
if 'tamper_message' not in st.session_state:
    st.session_state.tamper_message = None
if 'verification_result' not in st.session_state:
    st.session_state.verification_result = None

st.title("🛡️ ML Code Security Audit: Vulnerability Simulator")
st.markdown("""
**Interactive demonstration of Insecure Deserialization, Random State flaws & Unencrypted Model Storage.**  
*Compare the 'Vulnerable' implementation against the 'Secure' patch.*
""")


# --- Helper Functions ---
def get_file_status(filename):
    """Check if a file exists and return its status."""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        return f"✅ Created ({size:,} bytes)"
    return "❌ Not Found"


def get_file_hash(filename):
    """Calculate SHA-256 hash of a file."""
    if not os.path.exists(filename):
        return None
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_integrity(model_file, hash_file):
    """Verify model integrity by comparing hashes."""
    if not os.path.exists(model_file) or not os.path.exists(hash_file):
        return None, "Files not found"

    # Calculate actual hash
    actual_hash = get_file_hash(model_file)

    # Read expected hash
    with open(hash_file, 'r') as f:
        expected_hash = f.read().strip()

    if actual_hash == expected_hash:
        return True, actual_hash
    else:
        return False, f"Expected: {expected_hash[:24]}...\nGot: {actual_hash[:24]}..."


def check_encryption_key():
    """Check if encryption key exists."""
    if os.path.exists('encryption.key'):
        size = os.path.getsize('encryption.key')
        return f"✅ Present ({size} bytes)"
    return "❌ Not Generated"


# --- Main Interface ---
col1, col2 = st.columns(2)

# === COLUMN 1: VULNERABLE ===
with col1:
    st.header("❌ Vulnerable Code")
    st.error("**Risks:** Pickle RCE, No Input Validation, Fixed Random State, Unencrypted Storage")

    # 1. Show Code
    with st.expander("📄 View Source Code"):
        try:
            with open("vulnerable_code.py", "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        except FileNotFoundError:
            st.error("File 'vulnerable_code.py' not found.")

    # 2. Run Button
    if st.button("💀 Run Vulnerable Script", type="primary"):
        try:
            result = subprocess.run(
                [sys.executable, "vulnerable_code.py"],
                check=True,
                capture_output=True,
                text=True
            )
            st.toast("Vulnerable Script Executed!", icon="💀")
            with st.expander("📋 Execution Log", expanded=True):
                st.code(result.stdout, language="text")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Execution Failed: Script returned error code {e.returncode}")
            if e.stderr:
                st.code(e.stderr, language="text")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # 3. Artifact Check
    st.subheader("📦 Artifacts Generated")

    # Model file status
    model_status = get_file_status('model.pkl')
    st.write(f"**Insecure Pickle (`model.pkl`):** {model_status}")

    # Security warnings for vulnerable artifacts
    if os.path.exists('model.pkl'):
        st.warning("""
        ⚠️ **Security Issues:**
        - 🔓 Model saved as unencrypted pickle
        - 🎯 Vulnerable to Remote Code Execution (RCE)
        - 🔢 Uses predictable random_state=42
        - 📝 No integrity verification
        """)

# === COLUMN 2: SECURE ===
with col2:
    st.header("✅ Secure Implementation")
    st.success("**Fixes:** Joblib + Encryption, Input Sanitization, Secure Random, Integrity Hashing")

    # 1. Show Code
    with st.expander("📄 View Source Code"):
        try:
            with open("secure_code.py", "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        except FileNotFoundError:
            st.error("File 'secure_code.py' not found.")

    # 2. Run Button
    if st.button("🛡️ Run Secure Script"):
        try:
            result = subprocess.run(
                [sys.executable, "secure_code.py"],
                check=True,
                capture_output=True,
                text=True
            )
            st.toast("Secure Script Executed!", icon="🛡️")

            # Clear any previous tamper/verification messages
            st.session_state.tamper_message = None
            st.session_state.verification_result = None

            # Show execution output in expander
            with st.expander("📋 Execution Log", expanded=True):
                st.code(result.stdout, language="text")

            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Execution Failed: Script returned error code {e.returncode}")
            if e.stderr:
                st.code(e.stderr, language="text")
            if e.stdout:
                st.code(e.stdout, language="text")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # 3. Artifact Check
    st.subheader("📦 Artifacts Generated")

    # Encrypted model status
    encrypted_model = 'secure_model.encrypted'
    hash_file = 'secure_model.encrypted.sha256'
    key_file = 'encryption.key'

    st.write(f"**🔐 Encrypted Model (`{encrypted_model}`):** {get_file_status(encrypted_model)}")
    st.write(f"**🔏 Integrity Hash (`.sha256`):** {get_file_status(hash_file)}")
    st.write(f"**🔑 Encryption Key:** {check_encryption_key()}")

    # Security features confirmation
    if os.path.exists(encrypted_model):
        st.info("""
        🔒 **Security Features Applied:**
        - ✅ Model encrypted with Fernet (AES-128-CBC)
        - ✅ SHA-256 integrity hash generated
        - ✅ Cryptographically secure random seed
        - ✅ Input validation & sanitization
        """)

st.divider()

# --- Integrity Verification Section ---
st.subheader("🔍 Integrity Verification Tool")

# Show any persistent messages from session state
if st.session_state.tamper_message:
    st.error(st.session_state.tamper_message)

if st.session_state.verification_result:
    result_type, message = st.session_state.verification_result
    if result_type == "success":
        st.success(message)
    elif result_type == "error":
        st.error(message)
    elif result_type == "warning":
        st.warning(message)

verify_col1, verify_col2 = st.columns(2)

with verify_col1:
    st.markdown("**Verify Encrypted Model Integrity**")
    if st.button("🔎 Verify Secure Model"):
        result, message = verify_integrity(encrypted_model, hash_file)
        if result is None:
            st.session_state.verification_result = ("warning", f"⚠️ Cannot verify: {message}")
        elif result:
            st.session_state.verification_result = ("success",
                                                    f"✅ **INTEGRITY VERIFIED!**\n\nHash: `{message[:32]}...`")
            # Clear tamper message on successful verification after re-running secure script
            st.session_state.tamper_message = None
        else:
            st.session_state.verification_result = ("error",
                                                    f"❌ **INTEGRITY CHECK FAILED!**\n\n{message}\n\n🚨 The model may have been tampered with!")
        st.rerun()

with verify_col2:
    st.markdown("**Simulate Tampering Attack**")
    if st.button("💉 Tamper with Encrypted Model"):
        if os.path.exists(encrypted_model):
            # Get file size before tampering
            size_before = os.path.getsize(encrypted_model)

            # Append random bytes to simulate tampering
            with open(encrypted_model, 'ab') as f:
                f.write(b'\x00\x01\x02\x03TAMPERED_BY_ATTACKER')

            # Get file size after tampering
            size_after = os.path.getsize(encrypted_model)

            # Store message in session state so it persists after rerun
            st.session_state.tamper_message = f"""🚨 **MODEL FILE HAS BEEN TAMPERED WITH!**

**What happened:**
- Malicious bytes were appended to `{encrypted_model}`
- File size changed: {size_before:,} → {size_after:,} bytes (+{size_after - size_before} bytes)

**Next step:** Click "🔎 Verify Secure Model" to see the integrity check **FAIL**."""

            # Clear previous verification result
            st.session_state.verification_result = None

            st.rerun()
        else:
            st.warning("⚠️ Run the secure script first to generate the encrypted model.")

# Add a reset button
st.markdown("---")
if st.button("🔄 Reset Messages"):
    st.session_state.tamper_message = None
    st.session_state.verification_result = None
    st.rerun()

st.divider()

# --- Encryption Key Management Section ---
st.subheader("🔑 Encryption Key Management")

key_col1, key_col2, key_col3 = st.columns(3)

with key_col1:
    if st.button("👁️ View Key (Demo Only)"):
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                key = f.read()
            st.code(key.decode(), language="text")
            st.warning("⚠️ In production, NEVER display encryption keys!")
        else:
            st.info("No encryption key found. Run the secure script first.")

with key_col2:
    if st.button("🗑️ Delete Key (Test Recovery)"):
        if os.path.exists(key_file):
            os.remove(key_file)
            st.success("✅ Key deleted. Model is now unrecoverable without backup!")
            st.rerun()
        else:
            st.info("No key to delete.")

with key_col3:
    if st.button("🔄 Regenerate All"):
        # Delete all generated files
        files_to_delete = [key_file, encrypted_model, hash_file, 'model.pkl',
                           'secure_model.joblib', 'secure_model.joblib.sha256']
        deleted = []
        for f in files_to_delete:
            if os.path.exists(f):
                os.remove(f)
                deleted.append(f)

        st.session_state.tamper_message = None
        st.session_state.verification_result = None

        if deleted:
            st.success(f"✅ Deleted: {', '.join(deleted)}")
            st.info("Run secure script to generate new encrypted model with new key.")
        else:
            st.info("No files to delete.")
        st.rerun()

st.divider()

# --- Security Comparison Table ---
st.subheader("📊 Security Comparison")

comparison_data = {
    "Security Feature": [
        "Model Serialization",
        "Encryption",
        "Integrity Check",
        "Random State",
        "Input Validation",
        "RCE Vulnerability"
    ],
    "❌ Vulnerable": [
        "Pickle (insecure)",
        "None",
        "None",
        "Fixed (42)",
        "None",
        "HIGH RISK"
    ],
    "✅ Secure": [
        "Joblib (safer)",
        "Fernet AES-128",
        "SHA-256 Hash",
        "Cryptographic (secrets)",
        "Full sanitization",
        "Mitigated"
    ]
}

import pandas as pd

df = pd.DataFrame(comparison_data)
st.table(df)

st.divider()

# --- Audit Report Section ---
st.subheader("📝 Audit Findings")
try:
    with open("AUDIT_REPORT.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())
except FileNotFoundError:
    # Create default audit report
    default_report = """
## ML Security Audit Report

### Vulnerabilities Identified & Mitigated

| # | Vulnerability | Risk Level | Status |
|---|--------------|------------|--------|
| 1 | Lack of data validation | High | ✅ Fixed |
| 2 | No input validation | Medium | ✅ Fixed |
| 3 | Fixed random state | Medium | ✅ Fixed |
| 4 | Unencrypted model storage | High | ✅ Fixed |
| 5 | No integrity checks | High | ✅ Fixed |

### Recommendations
1. Store encryption keys in a secure key vault (AWS KMS, Azure Key Vault)
2. Implement key rotation policies
3. Add audit logging for model access
4. Consider model signing for additional authenticity verification
"""
    st.markdown(default_report)

# --- Footer ---
st.divider()
st.caption("🔐 ML Security Audit Tool | Demonstrates OWASP ML Security Best Practices")

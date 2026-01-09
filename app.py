import streamlit as st
import os
import sys  # <--- Added to fix execution path
import subprocess

# --- Page Config ---
st.set_page_config(page_title="ML Security Audit", page_icon="🔐", layout="wide")

st.title("🛡️ ML Code Security Audit: Vulnerability Simulator")
st.markdown("""
**Interactive demonstration of Insecure Deserialization & Random State flaws.**
*Compare the 'Vulnerable' implementation against the 'Secure' patch.*
""")


# --- Helper to Check Artifacts ---
def get_file_status(filename):
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        return f"✅ Created ({size} bytes)"
    return "❌ Not Found"


# --- Main Interface ---
col1, col2 = st.columns(2)

# === COLUMN 1: VULNERABLE ===
with col1:
    st.header("❌ Vulnerable Code")
    st.error("Risks: Pickle RCE, No Input Validation")

    # 1. Show Code (FIXED: Added encoding='utf-8')
    with st.expander("View Source Code"):
        try:
            with open("vulnerable_code.py", "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        except FileNotFoundError:
            st.error("File 'vulnerable_code.py' not found.")

    # 2. Run Button (FIXED: Uses sys.executable)
    if st.button("💀 Run Vulnerable Script", type="primary"):
        try:
            # Use sys.executable to ensure we use the same Python that has pandas installed
            subprocess.run([sys.executable, "vulnerable_code.py"], check=True)
            st.toast("Vulnerable Script Executed!", icon="💀")
            st.rerun()  # Refresh to show new artifacts
        except subprocess.CalledProcessError as e:
            st.error(f"Execution Failed: Script returned error code {e.returncode}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # 3. Artifact Check
    st.subheader("Artifacts Generated")
    st.write(f"**Insecure Pickle (`model.pkl`):** {get_file_status('model.pkl')}")

# === COLUMN 2: SECURE ===
with col2:
    st.header("✅ Secure Implementation")
    st.success("Fixes: Joblib + Hashing, Input Sanitization")

    # 1. Show Code (FIXED: Added encoding='utf-8')
    with st.expander("View Source Code"):
        try:
            with open("secure_code.py", "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        except FileNotFoundError:
            st.error("File 'secure_code.py' not found.")

    # 2. Run Button (FIXED: Uses sys.executable)
    if st.button("🛡️ Run Secure Script"):
        try:
            subprocess.run([sys.executable, "secure_code.py"], check=True)
            st.toast("Secure Script Executed!", icon="🛡️")
            st.rerun()  # Refresh to show new artifacts
        except subprocess.CalledProcessError as e:
            st.error(f"Execution Failed: Script returned error code {e.returncode}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # 3. Artifact Check
    st.subheader("Artifacts Generated")
    st.write(f"**Secure Model (`secure_model.joblib`):** {get_file_status('secure_model.joblib')}")
    st.write(f"**Integrity Hash (`.sha256`):** {get_file_status('secure_model.joblib.sha256')}")

st.divider()

# --- Audit Report Section ---
st.subheader("📝 Audit Findings")
try:
    with open("README.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())
except FileNotFoundError:
    st.warning("Audit Report not found.")
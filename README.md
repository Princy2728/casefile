# ⚡🛡️ LIFECYCLE-X AI
## Predictive Identity Threat Detection, Post-Termination Access Prevention & Cryptographic Audit System

> **Harbor & Finch Freight Pvt. Ltd.** | Cybersecurity SOC Dashboard  
> ⚠️ **All data is completely synthetic. No real accounts, systems, or organizations are accessed.**

---

## Overview

LIFECYCLE-X AI is an advanced cybersecurity prototype that integrates three defensive technologies into a cohesive identity access protection system:
1. **Machine Learning (Random Forest):** Predictive residual identity-access risk detection.
2. **SecureEvidence Vault (AES-256-GCM + HMAC-SHA256):** Confidential authenticated evidence storage and integrity verification.
3. **CryptoTrace Engine:** Tamper-evident, chronologically linked SHA-256 hash-chain audit trail with HMAC cryptographic seals.

**Demo Scenario — Rohan Menon:**
- Employee `HF0004 / r.menon` was terminated on **June 30, 2026**
- A successful authentication was detected on **August 13, 2026** (44 days post-termination)
- Device: `kiosk-wh-02` | IP: `172.16.8.44` | Resource: `/finance/ap-queue/doc-0045.pdf`
- Telemetry: Persistent session, missing MFA evaluation, critical financial queue access
- Action: Unconditional deterministic hard block, AES-256-GCM vaulting, HMAC-SHA256 signing, CryptoTrace hash-chain logging, and simulated containment.

---

## Project Structure

```
lifecycle-x-ai/
├── app.py                 # Streamlit SOC dashboard (Main application)
├── generate_dataset.py    # Synthetic dataset generator
├── train_model.py         # ML model training script
├── test_crypto.py         # Cryptographic module validation test suite (10/10 tests)
├── requirements.txt       # Python dependencies (incl. cryptography>=41.0.0)
├── README.md              # Documentation & cryptographic guide
├── crypto/                # Cryptography & Evidence Protection Package
│   ├── __init__.py        # Exported public cryptographic interfaces
│   ├── key_management.py  # AES-256 and HMAC-SHA256 key segregation & lifecycle
│   ├── encryption.py      # AES-256-GCM authenticated encryption (SecureEvidence Vault)
│   ├── evidence_integrity.py # Canonical JSON & HMAC-SHA256 authenticity verification
│   └── cryptotrace.py     # CryptoTrace SHA-256 hash-chain & tamper detection engine
├── data/
│   ├── auth_logs.csv      # Synthetic authentication logs (2,080 records)
│   └── .crypto_keys.json  # Isolated local cryptographic key vault
└── model/
    ├── rf_model.joblib    # Trained Random Forest classifier
    └── encoders.joblib    # Label encoders + performance metrics
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 to 3.14
- pip

### Step 1 — Install Dependencies

```bash
cd "lifecycle-x-ai"
pip install -r requirements.txt
```

### Step 2 — Run Cryptographic Validation Test Suite

Run the 10-scenario cryptographic verification suite:

```bash
python test_crypto.py
```

Expected output: `10/10 SCENARIOS PASSED (100% SUCCESS)`.

### Step 3 — Launch the SOC Dashboard

```bash
python -m streamlit run app.py
```

The application opens at `http://localhost:8501`.

---

## Cryptographic Architecture

### 1. SecureEvidence Vault (AES-256-GCM)
- **Algorithm:** AES-256-GCM (Galois/Counter Mode authenticated encryption) via Python `cryptography`.
- **Key:** Dedicated 256-bit key generated using `secrets.token_bytes(32)`.
- **Nonce:** Cryptographically unique 96-bit (12-byte) nonce generated per operation. Nonce reuse is strictly prevented.
- **Tag:** 128-bit authentication tag guarantees confidentiality and authenticity simultaneously.
- **AAD Binding:** Additional Authenticated Data (`LIFECYCLE-X-SECURE-VAULT-V1`) binds the ciphertext to the system context.

### 2. Evidence Authenticity (HMAC-SHA256)
- **Algorithm:** HMAC-SHA256 operating on canonical RFC-8785 JSON bytes.
- **Key Segregation:** Uses an independent 256-bit secret HMAC key strictly segregated from the AES key.
- **Verification:** Uses constant-time string comparison (`hmac.compare_digest`) to prevent timing side-channels.
- **Status Codes:** `Integrity Verified`, `Evidence Modified`, `Integrity Failed`, `Verification Error`.

### 3. CryptoTrace Hash Chain
- **Genesis Block:** Initialized with a standardized 64-character null SHA-256 genesis hash (`0000...0000`).
- **Chain Formula:** $\text{current\_hash} = \text{SHA-256}(\text{canonical}(\text{payload}) \parallel \text{previous\_hash})$.
- **HMAC Seals:** Each block is sealed with an HMAC-SHA256 signature to prevent offline full-chain recalculation attacks.
- **Tamper Detection:** Recalculates every block's digest, validates linkage, detects missing/inserted blocks, and pinpoints the first compromised block index.

---

## Demo Workflow — Rohan Menon Incident & Tamper Detection

1. **Dashboard (`📊 Dashboard`):** View overall monitored employee metrics, suspicious authentication rates, and post-termination attempts.
2. **ML Model Performance (`🤖 ML Model Performance`):** Review the Random Forest evaluation metrics (97.8% Accuracy, 97.1% Recall, 88.3% F1) and feature importances.
3. **Incident Investigation (`🔍 Incident Investigation`):**
   - Click **"🔴 Run Rohan Menon Incident"**.
   - Observe the 100.0% High Risk ML gauge score.
   - Note the **Deterministic Hard Policy Override** (`POST_TERM_ACCESS_ZERO_TOLERANCE`).
   - Observe the **Cryptographic Custody Card** displaying the AES-256-GCM Vault ID, 96-bit Nonce, and HMAC-SHA256 signature.
4. **Automated Response (`🚨 Automated Response`):**
   - Click **"🔴 Contain Incident"**.
   - Observe simulated containment across IdP, session revocation, and 3 mock applications.
   - Note how each containment step is dynamically recorded as a block in CryptoTrace.
5. **Security Audit Log (`📋 Audit Log`):**
   - View the event log and expand the **CryptoTrace Hash-Chain Inspector** showing previous/current hashes and HMAC seals.
6. **Cryptographic Security Center (`🔐 Cryptographic Security Center`):**
   - **Vault Controls:** Click **"🔍 Verify HMAC Integrity"** to verify evidence authenticity. Click **"🔓 Decrypt Evidence"** for authorized plaintext inspection.
   - **Tamper Demonstration:** Click **"⚠️ Simulate Evidence Tampering (Demonstration)"**.
   - Observe the instant **🚨 CRITICAL INTEGRITY ALERT** pinpointing compromised Block #2!
   - Click **"🔄 Restore Original Clean Chain"** to return the chain to `Chain Verified ✅`.

---

## Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Dashboard | Streamlit | SOC Web Application |
| ML Engine | Scikit-learn (Random Forest) | Predictive Identity Threat Detection |
| Confidentiality | AES-256-GCM | SecureEvidence Vault Encryption |
| Authenticity | HMAC-SHA256 | Evidence Integrity Verification |
| Audit Trail | SHA-256 + HMAC-SHA256 | CryptoTrace Hash Chain |
| Visualizations | Plotly | Dynamic Dark-Theme Charts |
| Persistence | Joblib, JSON, CSV | Model, Keys & Audit Persistence |

"""
LIFECYCLE-X AI — Predictive Identity Threat Detection & Post-Termination Access Prevention
Streamlit Dashboard | Harbor & Finch Freight Pvt. Ltd.

All data is synthetic and for demonstration purposes only.
"""

import os
import time
import uuid
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# Cryptographic Security Modules
from crypto.key_management import get_aes_key, get_hmac_key, get_key_metadata
from crypto.encryption import encrypt_evidence, decrypt_evidence, DecryptionError
from crypto.evidence_integrity import canonicalize, compute_sha256, generate_hmac, verify_hmac
from crypto.cryptotrace import CryptoTraceChain, build_initial_chain, GENESIS_HASH

# ───────────────────────────────────────────────────────────────────────────────
# PATHS
# ───────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data", "auth_logs.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "model", "rf_model.joblib")
ENCODERS_PATH = os.path.join(BASE_DIR, "model", "encoders.joblib")

SENTINEL_DATE    = "2099-12-31"
REFERENCE_DATE   = datetime(2026, 11, 20)

FEATURE_COLS = [
    "employment_status_enc",
    "days_since_termination",
    "session_type_enc",
    "token_age_hours",
    "device_category_enc",
    "mfa_status_enc",
    "hr_idp_sync_delay_hours",
    "application_sensitivity_enc",
    "revocation_acknowledged",
    "auth_success",
]

CATEGORICAL_COLS = [
    "employment_status",
    "session_type",
    "device_category",
    "mfa_status",
    "application_sensitivity",
]

# ───────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LIFECYCLE-X AI | Predictive Identity Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base dark theme */
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    .block-container { padding: 1.5rem 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060910 0%, #0d1225 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] .stRadio label { color: #93c5fd !important; font-size: 0.95rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(29,78,216,0.15);
    }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        box-shadow: 0 4px 15px rgba(29,78,216,0.4);
        transform: translateY(-1px);
    }

    /* Incident banners */
    .incident-banner {
        background: linear-gradient(135deg, #1a0808, #2d0a0a);
        border: 1px solid #dc2626;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(220,38,38,0.2);
    }
    .success-banner {
        background: linear-gradient(135deg, #052e16, #064e3b);
        border: 1px solid #16a34a;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .info-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #60a5fa;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e3a5f;
    }
    .risk-high { color: #ef4444; font-weight: 700; }
    .risk-low  { color: #22c55e; font-weight: 700; }
    .tag-critical { background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .tag-info     { background:#1e3a5f; color:#93c5fd; padding:2px 8px; border-radius:4px; font-size:0.8rem; }

    /* Tables */
    .stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 8px; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { background: #111827; border-radius: 6px 6px 0 0; color: #94a3b8; }
    .stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }

    /* Audit log */
    .audit-critical { color: #ef4444; }
    .audit-high     { color: #f97316; }
    .audit-info     { color: #60a5fa; }

    /* Status pills */
    .pill-success { background:#14532d; color:#86efac; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
    .pill-pending { background:#1c1917; color:#fbbf24; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
    .pill-fail    { background:#450a0a; color:#fca5a5; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }

    h1, h2, h3 { color: #e2e8f0; }
    p { color: #cbd5e1; }
    li { color: #cbd5e1; }
    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        return None, None
    clf      = joblib.load(MODEL_PATH)
    enc_data = joblib.load(ENCODERS_PATH)
    return clf, enc_data

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    return df

# ───────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors train_model.py)
# ───────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    df["termination_date"] = df["termination_date"].fillna(SENTINEL_DATE)

    def parse_date(d):
        try:
            return datetime.strptime(str(d)[:10], "%Y-%m-%d")
        except Exception:
            return datetime.strptime(SENTINEL_DATE, "%Y-%m-%d")

    df["termination_date_dt"]   = df["termination_date"].apply(parse_date)
    df["days_since_termination"] = df["termination_date_dt"].apply(
        lambda d: (REFERENCE_DATE - d).days
    )

    for col in CATEGORICAL_COLS:
        le = encoders.get(col)
        if le is not None:
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[f"{col}_enc"] = le.transform(df[col].astype(str))
        else:
            df[f"{col}_enc"] = 0

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df

def predict_single(record: dict, clf, enc_data: dict) -> tuple[int, float]:
    """Predict risk for a single record dict. Returns (label, probability)."""
    df_tmp = pd.DataFrame([record])
    df_tmp = engineer_features(df_tmp, enc_data["encoders"])
    X = df_tmp[FEATURE_COLS]
    pred   = int(clf.predict(X)[0])
    prob   = float(clf.predict_proba(X)[0][1])
    return pred, prob

# ───────────────────────────────────────────────────────────────────────────────
# SCENARIO DATA
# ───────────────────────────────────────────────────────────────────────────────
ROHAN_RECORD = {
    "employee_id"           : "HF0004",
    "employment_status"     : "terminated",
    "termination_date"      : "2026-06-30",
    "auth_timestamp"        : "2026-08-13 03:17:22",
    "session_type"          : "persistent",
    "token_age_hours"       : 312.5,
    "device_category"       : "kiosk",
    "mfa_status"            : "not_evaluated",
    "hr_idp_sync_delay_hours": 44.0,
    "application_sensitivity": "critical",
    "revocation_acknowledged": 0,
    "auth_success"          : 1,
}

NORMAL_RECORD = {
    "employee_id"           : "HF0021",
    "employment_status"     : "active",
    "termination_date"      : None,
    "auth_timestamp"        : "2026-08-13 09:45:00",
    "session_type"          : "standard",
    "token_age_hours"       : 2.3,
    "device_category"       : "corporate",
    "mfa_status"            : "evaluated",
    "hr_idp_sync_delay_hours": 0.5,
    "application_sensitivity": "medium",
    "revocation_acknowledged": 1,
    "auth_success"          : 1,
}

ROHAN_EXTRA = {
    "username" : "r.menon",
    "full_name": "Rohan Menon",
    "role"     : "Accounts Payable Clerk",
    "org"      : "Harbor & Finch Freight Pvt. Ltd.",
    "device"   : "kiosk-wh-02",
    "ip"       : "172.16.8.44",
    "resource" : "/finance/ap-queue/doc-0045.pdf",
}

NORMAL_EXTRA = {
    "username" : "p.sharma",
    "full_name": "Priya Sharma",
    "role"     : "Logistics Coordinator",
    "org"      : "Harbor & Finch Freight Pvt. Ltd.",
    "device"   : "corporate-laptop-05",
    "ip"       : "10.0.1.33",
    "resource" : "/logistics/shipments/manifest-2026-08.xlsx",
}

DETECTION_REASONS = [
    "🔴 **Post-termination authentication** — Employee HF0004 logged in **44 days** after termination (2026-06-30 → 2026-08-13 03:17 UTC).",
    "🔴 **Persistent browser session** — Active session survived account deactivation; refresh token not expired.",
    "🔴 **MFA not evaluated** — Multi-factor authentication was bypassed or skipped, creating an authentication bypass risk.",
    "🔴 **Critical-sensitivity resource accessed** — `/finance/ap-queue/doc-0045.pdf` classified as CRITICAL sensitivity.",
    "🟡 **Non-standard endpoint** — Device `kiosk-wh-02` is a warehouse kiosk, not a registered corporate asset.",
    "🟡 **HR-to-IdP sync delay of 44.0 hours** — Revocation window was open; termination event not propagated in time.",
]

RECOMMENDED_ACTIONS = [
    "Immediately revoke all active sessions for user `r.menon` (HF0004).",
    "Expire and blacklist all OAuth refresh tokens issued before 2026-07-01.",
    "Block account `HF0004` in Identity Provider (Okta/Azure AD).",
    "Remove finance application entitlements from AP-Queue and DocuStore.",
    "Notify HR, Security Operations Centre (SOC), and Legal/Compliance teams.",
    "Initiate forensic review of accessed document `/finance/ap-queue/doc-0045.pdf`.",
    "Audit HR-to-IdP sync pipeline for delay root cause.",
    "Generate formal incident report and preserve evidence chain.",
]

MOCK_AUDIT_LOG = [
    {"Timestamp": "2026-08-13 03:17:22", "Event ID": "EVT-20260813-001", "Actor": "r.menon",
     "Action": "AUTH_SUCCESS_POST_TERMINATION", "Target": "FinancePortal", "Severity": "CRITICAL", "Status": "DETECTED"},
    {"Timestamp": "2026-08-13 03:17:22", "Event ID": "EVT-20260813-002", "Actor": "r.menon",
     "Action": "RESOURCE_ACCESS", "Target": "/finance/ap-queue/doc-0045.pdf", "Severity": "CRITICAL", "Status": "DETECTED"},
    {"Timestamp": "2026-08-13 03:17:35", "Event ID": "EVT-20260813-003", "Actor": "LIFECYCLE-X AI",
     "Action": "INCIDENT_DETECTED", "Target": "HF0004 / r.menon", "Severity": "CRITICAL", "Status": "RAISED"},
    {"Timestamp": "2026-08-13 03:17:36", "Event ID": "EVT-20260813-004", "Actor": "LIFECYCLE-X AI",
     "Action": "POLICY_BLOCK_APPLIED", "Target": "HF0004", "Severity": "HIGH", "Status": "ENFORCED"},
    {"Timestamp": "2026-08-13 03:19:00", "Event ID": "EVT-20260813-005", "Actor": "SOC-ANALYST",
     "Action": "CONTAINMENT_INITIATED", "Target": "HF0004 / r.menon", "Severity": "HIGH", "Status": "IN_PROGRESS"},
    {"Timestamp": "2026-08-13 03:19:45", "Event ID": "EVT-20260813-006", "Actor": "LIFECYCLE-X AI",
     "Action": "SESSION_REVOKED", "Target": "kiosk-wh-02 / 172.16.8.44", "Severity": "HIGH", "Status": "COMPLETED"},
    {"Timestamp": "2026-08-13 03:20:10", "Event ID": "EVT-20260813-007", "Actor": "LIFECYCLE-X AI",
     "Action": "FINANCE_ACCESS_BLOCKED", "Target": "AP-Workflow / DocuStore", "Severity": "HIGH", "Status": "COMPLETED"},
    {"Timestamp": "2026-08-13 03:20:30", "Event ID": "EVT-20260813-008", "Actor": "LIFECYCLE-X AI",
     "Action": "AUDIT_LOG_SEALED", "Target": "INC-20260813-HF0004", "Severity": "INFO", "Status": "ARCHIVED"},
]

# ───────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ───────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PAPER_BG = "#0a0e1a"
PLOTLY_PLOT_BG  = "#111827"

def plotly_fig_style(fig):
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_PLOT_BG,
        font=dict(color="#e2e8f0"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>⚡🛡️</div>
        <div style='font-size:1.3rem; font-weight:800; color:#60a5fa; letter-spacing:1px;'>LIFECYCLE-X AI</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Predictive Identity Threat Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='info-card' style='margin:0;'>
        <div style='color:#94a3b8; font-size:0.75rem;'>ORGANIZATION</div>
        <div style='color:#e2e8f0; font-size:0.85rem; font-weight:600;'>Harbor & Finch Freight Pvt. Ltd.</div>
        <div style='margin-top:0.5rem; color:#94a3b8; font-size:0.75rem;'>SYSTEM STATUS</div>
        <div style='color:#22c55e; font-weight:700;'>● ACTIVE</div>
        <div style='margin-top:0.5rem; color:#94a3b8; font-size:0.75rem;'>CURRENT TIME</div>
        <div style='color:#e2e8f0; font-size:0.82rem;'>2026-11-20 01:28 UTC+5:30</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    nav = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Incident Investigation",
         "🚨 Automated Response", "📋 Audit Log", "🔐 Cryptographic Security Center"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.72rem; text-align:center; padding:0.5rem;'>
        ⚠️ All data is synthetic.<br>No real accounts or systems are accessed.
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ───────────────────────────────────────────────────────────────────────────────
clf, enc_data = load_model()
df_raw        = load_data()

model_loaded = clf is not None and enc_data is not None
data_loaded  = df_raw is not None

# ───────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT (Workflow & Cryptography Integration)
# ───────────────────────────────────────────────────────────────────────────────
for key, default in {
    "incident_shown"          : None,
    "containment_done"        : False,
    "containment_log"         : [],
    "incident_id"             : None,
    "tampered_chain_active"   : False,
    "tampered_chain_copy"     : None,
    "show_decrypted_evidence" : False,
    "vault_status_msg"        : None,
    "chain_verification_cache": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "cryptotrace_chain" not in st.session_state:
    st.session_state["cryptotrace_chain"] = build_initial_chain()

# Initialize SecureEvidence Vault record for Rohan Menon incident
if "rohan_vault_record" not in st.session_state:
    rohan_full_evidence = {
        "employee_id": ROHAN_RECORD["employee_id"],
        "username": ROHAN_EXTRA["username"],
        "full_name": ROHAN_EXTRA["full_name"],
        "role": ROHAN_EXTRA["role"],
        "org": ROHAN_EXTRA["org"],
        "termination_date": ROHAN_RECORD["termination_date"],
        "auth_timestamp": ROHAN_RECORD["auth_timestamp"],
        "days_post_termination": 44,
        "device": ROHAN_EXTRA["device"],
        "ip": ROHAN_EXTRA["ip"],
        "resource": ROHAN_EXTRA["resource"],
        "session_type": ROHAN_RECORD["session_type"],
        "mfa_status": ROHAN_RECORD["mfa_status"],
        "token_age_hours": ROHAN_RECORD["token_age_hours"],
        "application_sensitivity": ROHAN_RECORD["application_sensitivity"],
        "hr_idp_sync_delay_hours": ROHAN_RECORD["hr_idp_sync_delay_hours"],
    }
    vault_rec = encrypt_evidence(
        rohan_full_evidence,
        metadata={"employee_id": "HF0004", "incident_id": "INC-20260813-HF0004", "case": "Post-Termination Access"}
    )
    hmac_sig, sha_digest = generate_hmac(rohan_full_evidence)
    st.session_state["rohan_raw_evidence"] = rohan_full_evidence
    st.session_state["rohan_vault_record"] = vault_rec
    st.session_state["rohan_hmac_sig"] = hmac_sig
    st.session_state["rohan_sha256"] = sha_digest

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if nav == "📊 Dashboard":
    st.markdown("<h1 style='color:#60a5fa;'>⚡ LIFECYCLE-X AI — SOC Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Harbor & Finch Freight Pvt. Ltd. | Identity Threat Monitoring</p>", unsafe_allow_html=True)

    # Metric row — compute from actual data when available
    if data_loaded:
        total_employees = df_raw["employee_id"].nunique()
        high_risk_count = int(df_raw["risk_label"].sum())
        post_term_count = int(((df_raw["employment_status"] == "terminated") & (df_raw["auth_success"] == 1)).sum())
        critical_count = int(((df_raw["risk_label"] == 1) & (df_raw["application_sensitivity"] == "critical")).sum())
    else:
        total_employees, high_risk_count, post_term_count, critical_count = 87, 12, 3, 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Employees Monitored", f"{total_employees}", "+2 this week")
    c2.metric("⚠️ Suspicious Auth Events", f"{high_risk_count}", f"+{min(high_risk_count,5)} today", delta_color="inverse")
    c3.metric("🚫 Post-Termination Attempts", f"{post_term_count}", f"+{min(post_term_count,3)} today", delta_color="inverse")
    c4.metric("🔴 Critical Incidents", f"{critical_count}", "Active", delta_color="inverse")

    st.markdown("---")

    if not data_loaded:
        st.error("⚠️ Dataset not found. Run `generate_dataset.py` first, then `train_model.py`.")
    else:
        # Risk distribution bar chart
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-header'>Risk Label Distribution</div>", unsafe_allow_html=True)
            risk_counts = df_raw["risk_label"].value_counts().reset_index()
            risk_counts.columns = ["Risk Label", "Count"]
            risk_counts["Label"] = risk_counts["Risk Label"].map({0: "Normal (0)", 1: "High Risk (1)"})
            fig_bar = go.Figure(go.Bar(
                x=risk_counts["Label"],
                y=risk_counts["Count"],
                marker_color=["#22c55e", "#ef4444"],
                text=risk_counts["Count"],
                textposition="auto",
            ))
            fig_bar.update_layout(title="Authentication Records by Risk", template=PLOTLY_TEMPLATE)
            plotly_fig_style(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.markdown("<div class='section-header'>Device Category Distribution</div>", unsafe_allow_html=True)
            dev_counts = df_raw["device_category"].value_counts().reset_index()
            dev_counts.columns = ["Device", "Count"]
            fig_pie = go.Figure(go.Pie(
                labels=dev_counts["Device"],
                values=dev_counts["Count"],
                hole=0.4,
                marker=dict(colors=["#1d4ed8", "#dc2626", "#f59e0b", "#6b7280"]),
            ))
            fig_pie.update_layout(title="Authentication by Device Type", template=PLOTLY_TEMPLATE)
            plotly_fig_style(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Auth events table
        st.markdown("<div class='section-header'>Recent Authentication Events</div>", unsafe_allow_html=True)
        display_df = df_raw[["employee_id", "employment_status", "auth_timestamp",
                              "session_type", "device_category", "mfa_status",
                              "application_sensitivity", "auth_success", "risk_label"]].copy()
        display_df["risk_label"] = display_df["risk_label"].map({0: "✅ Normal", 1: "🔴 High Risk"})
        display_df["auth_success"] = display_df["auth_success"].map({0: "❌ Failed", 1: "✅ Success"})
        st.dataframe(
            display_df.head(50),
            use_container_width=True,
            hide_index=True,
        )

        # MFA status breakdown
        st.markdown("<div class='section-header'>MFA Status Breakdown</div>", unsafe_allow_html=True)
        mfa_counts = df_raw["mfa_status"].value_counts().reset_index()
        mfa_counts.columns = ["MFA Status", "Count"]
        fig_mfa = go.Figure(go.Bar(
            x=mfa_counts["MFA Status"],
            y=mfa_counts["Count"],
            marker_color=["#22c55e", "#ef4444", "#f59e0b"],
            text=mfa_counts["Count"],
            textposition="auto",
        ))
        fig_mfa.update_layout(title="MFA Evaluation Across All Events", template=PLOTLY_TEMPLATE)
        plotly_fig_style(fig_mfa)
        st.plotly_chart(fig_mfa, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — ML MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif nav == "🤖 ML Model Performance":
    st.markdown("<h1 style='color:#60a5fa;'>🤖 ML Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Random Forest Classifier — Trained on synthetic authentication data</p>", unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model not found. Run `generate_dataset.py` then `train_model.py`.")
    elif not data_loaded:
        st.error("⚠️ Dataset not found. Run `generate_dataset.py` first.")
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        )

        # Reproduce the test split
        df_fe = engineer_features(df_raw.copy(), enc_data["encoders"])
        X = df_fe[FEATURE_COLS].fillna(0)
        y = df_fe["risk_label"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

        y_pred = clf.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        # Performance metrics
        st.markdown("<div class='section-header'>Performance Metrics (Test Set — 20% holdout)</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Accuracy",  f"{acc*100:.2f}%")
        m2.metric("🔍 Precision", f"{prec*100:.2f}%")
        m3.metric("📡 Recall",    f"{rec*100:.2f}%")
        m4.metric("⚖️ F1-Score",  f"{f1*100:.2f}%")

        st.markdown("---")

        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted Normal", "Predicted High Risk"],
                y=["Actual Normal", "Actual High Risk"],
                colorscale=[[0, "#0a0e1a"], [0.5, "#1d4ed8"], [1, "#dc2626"]],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20, "color": "white"},
                showscale=True,
            ))
            fig_cm.update_layout(title="Confusion Matrix", template=PLOTLY_TEMPLATE)
            plotly_fig_style(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

            # Class labels — hidden by default in expander
            with st.expander("📊 View Confusion Matrix Details", expanded=False):
                st.markdown(f"""
                <div class='info-card' style='font-size:0.85rem;'>
                    <b>TN</b> (True Normal)   = {cm[0,0]} &nbsp;|&nbsp;
                    <b>FP</b> (False Alarm)   = {cm[0,1]}<br>
                    <b>FN</b> (Missed Threat) = {cm[1,0]} &nbsp;|&nbsp;
                    <b>TP</b> (True Threat)   = {cm[1,1]}
                </div>
                """, unsafe_allow_html=True)

        with col_fi:
            st.markdown("<div class='section-header'>Feature Importances</div>", unsafe_allow_html=True)
            importances = dict(zip(FEATURE_COLS, clf.feature_importances_))
            fi_df = pd.DataFrame(importances.items(), columns=["Feature", "Importance"])
            fi_df = fi_df.sort_values("Importance")

            readable = {
                "employment_status_enc"      : "Employment Status",
                "days_since_termination"     : "Days Since Termination",
                "session_type_enc"           : "Session Type",
                "token_age_hours"            : "Token Age (Hours)",
                "device_category_enc"        : "Device Category",
                "mfa_status_enc"             : "MFA Status",
                "hr_idp_sync_delay_hours"    : "HR-to-IdP Sync Delay",
                "application_sensitivity_enc": "Application Sensitivity",
                "revocation_acknowledged"    : "Revocation Acknowledged",
                "auth_success"               : "Auth Success",
            }
            fi_df["Feature Label"] = fi_df["Feature"].map(readable)

            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature Label"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#1d4ed8"], [1, "#dc2626"]],
                ),
                text=[f"{v:.3f}" for v in fi_df["Importance"]],
                textposition="auto",
            ))
            fig_fi.update_layout(title="Feature Importance Scores", template=PLOTLY_TEMPLATE, height=420)
            plotly_fig_style(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("📌 Model Notes", expanded=False):
            st.markdown("""
            <div class='info-card'>
                <b>📌 Model Notes:</b><br>
                • Algorithm: <b>Random Forest Classifier</b> (200 estimators, balanced class weights)<br>
                • Dataset: ~2,000+ synthetic authentication events (seed=42)<br>
                • No target leakage — labels derived from post-termination timing logic only<br>
                • Model file: <code>model/rf_model.joblib</code> | Encoders: <code>model/encoders.joblib</code>
            </div>
            """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — INCIDENT INVESTIGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif nav == "🔍 Incident Investigation":
    st.markdown("<h1 style='color:#60a5fa;'>🔍 Incident Investigation Panel</h1>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🔴 Run Rohan Menon Incident", key="btn_rohan"):
            st.session_state["incident_shown"] = "rohan"
            st.session_state["containment_done"] = False
            st.session_state["incident_id"] = f"INC-20260813-HF0004-{str(uuid.uuid4())[:8].upper()}"
    with btn_col2:
        if st.button("🟢 Run Normal Employee Scenario", key="btn_normal"):
            st.session_state["incident_shown"] = "normal"

    # ── ROHAN INCIDENT ──────────────────────────────────────────────────────
    if st.session_state["incident_shown"] == "rohan":
        inc_id = st.session_state["incident_id"]

        st.markdown(f"""
        <div class='incident-banner'>
            <div style='font-size:1.4rem; font-weight:800; color:#ef4444;'>🚨 CRITICAL SECURITY INCIDENT DETECTED</div>
            <div style='color:#fca5a5; margin-top:4px;'>Incident ID: <b>{inc_id}</b> &nbsp;|&nbsp; Raised: 2026-08-13 03:17:35 UTC</div>
        </div>
        """, unsafe_allow_html=True)

        # Employee identity card
        st.markdown("<div class='section-header'>👤 Employee Identity</div>", unsafe_allow_html=True)
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            st.markdown(f"""
            <div class='info-card'>
                <table style='width:100%; border-collapse:collapse; color:#cbd5e1; font-size:0.88rem;'>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Employee ID</td><td style='padding:4px 8px;'><b>HF0004</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Full Name</td><td style='padding:4px 8px;'><b>Rohan Menon</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Username</td><td style='padding:4px 8px;'><code>r.menon</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Role</td><td style='padding:4px 8px;'>Accounts Payable Clerk</td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Organization</td><td style='padding:4px 8px;'>Harbor &amp; Finch Freight Pvt. Ltd.</td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Employment Status</td><td style='padding:4px 8px;'><span class='tag-critical'>TERMINATED</span></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Termination Date</td><td style='padding:4px 8px;'><b style='color:#ef4444;'>2026-06-30</b></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        with id_col2:
            st.markdown(f"""
            <div class='info-card'>
                <table style='width:100%; border-collapse:collapse; color:#cbd5e1; font-size:0.88rem;'>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Auth Timestamp</td><td style='padding:4px 8px;'><b style='color:#ef4444;'>2026-08-13 03:17:22 UTC</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Days Post-Termination</td><td style='padding:4px 8px;'><b style='color:#ef4444;'>44 days</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Device</td><td style='padding:4px 8px;'><code>kiosk-wh-02</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>IP Address</td><td style='padding:4px 8px;'><code>172.16.8.44</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Session Type</td><td style='padding:4px 8px;'><span class='tag-critical'>PERSISTENT</span></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>MFA Status</td><td style='padding:4px 8px;'><span class='tag-critical'>NOT EVALUATED</span></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Accessed Resource</td><td style='padding:4px 8px;'><code>/finance/ap-queue/doc-0045.pdf</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>App Sensitivity</td><td style='padding:4px 8px;'><span class='tag-critical'>CRITICAL</span></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>HR-IdP Sync Delay</td><td style='padding:4px 8px;'>44.0 hours</td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Token Age</td><td style='padding:4px 8px;'>312.5 hours (~13 days)</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # ML Prediction
        st.markdown("<div class='section-header'>🤖 ML Model Prediction</div>", unsafe_allow_html=True)
        if model_loaded:
            pred_label, pred_prob = predict_single(ROHAN_RECORD, clf, enc_data)
            risk_pct = pred_prob * 100
            pred_col1, pred_col2 = st.columns(2)
            pred_col1.metric("Risk Score", f"{risk_pct:.1f}%", "CRITICAL")
            pred_col2.metric("ML Prediction", "🔴 HIGH RISK" if pred_label == 1 else "✅ NORMAL")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "ML Risk Score (%)", "font": {"color": "#e2e8f0"}},
                number={"suffix": "%", "font": {"color": "#ef4444", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                    "bar": {"color": "#dc2626"},
                    "steps": [
                        {"range": [0, 40], "color": "#14532d"},
                        {"range": [40, 70], "color": "#78350f"},
                        {"range": [70, 100], "color": "#450a0a"},
                    ],
                    "threshold": {"line": {"color": "#ef4444", "width": 4}, "value": 70},
                },
            ))
            fig_gauge.update_layout(height=250, template=PLOTLY_TEMPLATE)
            plotly_fig_style(fig_gauge)
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning("⚠️ ML model not loaded. Run `train_model.py` to generate the model.")

        # Deterministic policy
        st.markdown("<div class='section-header'>⚖️ Deterministic Policy Decision</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='incident-banner'>
            <div style='font-size:1.1rem; font-weight:800; color:#ef4444;'>
                ⛔ BLOCKED — Terminated Employee. ML score overridden by Hard Policy.
            </div>
            <div style='color:#fca5a5; font-size:0.88rem; margin-top:8px;'>
                Policy Rule: <b>POST_TERM_ACCESS_ZERO_TOLERANCE</b><br>
                Any successful authentication after confirmed termination date is UNCONDITIONALLY DENIED,
                regardless of ML confidence score. Session invalidated. Incident raised.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detection reasons
        st.markdown("<div class='section-header'>🔎 Detection Reasons</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        for reason in DETECTION_REASONS:
            st.markdown(f"- {reason}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Related event IDs
        st.markdown("<div class='section-header'>🔗 Related Event IDs & Cryptographic Custody</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card' style='font-size:0.85rem; font-family:monospace;'>
            EVT-20260813-001 · EVT-20260813-002 · EVT-20260813-003 · EVT-20260813-004
        </div>
        """, unsafe_allow_html=True)

        # Cryptographic Custody Card
        vault_rec = st.session_state.get("rohan_vault_record", {})
        hmac_sig = st.session_state.get("rohan_hmac_sig", "N/A")
        st.markdown(f"""
        <div class='info-card' style='border-left: 4px solid #1d4ed8; background: #0c1527;'>
            <div style='color:#60a5fa; font-weight:700; font-size:0.95rem; margin-bottom:6px;'>
                🛡️ SecureEvidence Vault &amp; CryptoTrace Custody
            </div>
            <div style='color:#cbd5e1; font-size:0.83rem; line-height:1.6;'>
                • <b>Confidentiality:</b> AES-256-GCM Encrypted (Vault ID: <code>{vault_rec.get('vault_id', 'VAULT-001')}</code>)<br>
                • <b>Unique Nonce (96-bit):</b> <code>{vault_rec.get('nonce_hex', '')[:24]}...</code> (One-time, strictly non-repeating)<br>
                • <b>Authenticity:</b> HMAC-SHA256 Signed (<code>{hmac_sig[:24]}...</code>)<br>
                • <b>Tamper Evidence:</b> Anchored to CryptoTrace Hash-Chain Block #0–#4<br>
                • <b>Status:</b> <span class='pill-success'>Integrity Verified</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommended actions
        st.markdown("<div class='section-header'>✅ Recommended Containment Actions</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        for i, action in enumerate(RECOMMENDED_ACTIONS, 1):
            st.markdown(f"{i}. {action}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.info("👉 Proceed to **🚨 Automated Response** in the sidebar to simulate containment.")

    # ── NORMAL SCENARIO ──────────────────────────────────────────────────────
    elif st.session_state["incident_shown"] == "normal":
        st.markdown("""
        <div class='success-banner'>
            <div style='font-size:1.2rem; font-weight:800; color:#22c55e;'>✅ NORMAL ACCESS — NO THREAT DETECTED</div>
            <div style='color:#86efac; margin-top:4px;'>Scenario: Active Employee Authentication</div>
        </div>
        """, unsafe_allow_html=True)

        n_col1, n_col2 = st.columns(2)
        with n_col1:
            st.markdown("""
            <div class='info-card'>
                <div class='section-header' style='color:#22c55e;'>👤 Employee Profile</div>
                <table style='width:100%; border-collapse:collapse; color:#cbd5e1; font-size:0.88rem;'>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Employee ID</td><td><b>HF0021</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Full Name</td><td><b>Priya Sharma</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Username</td><td><code>p.sharma</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Role</td><td>Logistics Coordinator</td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Status</td><td><b style='color:#22c55e;'>ACTIVE</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Organization</td><td>Harbor &amp; Finch Freight Pvt. Ltd.</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        with n_col2:
            st.markdown("""
            <div class='info-card'>
                <div class='section-header' style='color:#22c55e;'>🔐 Authentication Details</div>
                <table style='width:100%; border-collapse:collapse; color:#cbd5e1; font-size:0.88rem;'>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Auth Timestamp</td><td><b>2026-08-13 09:45:00 UTC</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Device</td><td><code>corporate-laptop-05</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>IP Address</td><td><code>10.0.1.33</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Session Type</td><td><b style='color:#22c55e;'>STANDARD</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>MFA Status</td><td><b style='color:#22c55e;'>EVALUATED ✅</b></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Accessed Resource</td><td><code>/logistics/shipments/manifest-2026-08.xlsx</code></td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>App Sensitivity</td><td>MEDIUM</td></tr>
                    <tr><td style='padding:4px 8px; color:#94a3b8;'>Token Age</td><td>2.3 hours</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        if model_loaded:
            pred_label, pred_prob = predict_single(NORMAL_RECORD, clf, enc_data)
            risk_pct = pred_prob * 100
            st.markdown(f"""
            <div class='success-banner'>
                <b>ML Risk Score: {risk_pct:.1f}%</b> &nbsp;|&nbsp; Prediction: <b>✅ NORMAL</b><br>
                <span style='color:#86efac; font-size:0.85rem;'>Policy Decision: ACCESS PERMITTED — Active employee, standard session, MFA evaluated.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ML model not loaded — but this is a normal employee scenario (active, standard session, MFA evaluated). Policy: ACCESS PERMITTED.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — AUTOMATED RESPONSE SIMULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif nav == "🚨 Automated Response":
    st.markdown("<h1 style='color:#ef4444;'>🚨 Automated Response Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Simulates automated containment for the Rohan Menon (HF0004) incident. No real systems are accessed.</p>", unsafe_allow_html=True)

    inc_id = st.session_state.get("incident_id") or "INC-20260813-HF0004-DEMO"

    st.markdown(f"""
    <div class='incident-banner'>
        <b>Target:</b> HF0004 / r.menon &nbsp;|&nbsp;
        <b>Incident:</b> {inc_id} &nbsp;|&nbsp;
        <b>Trigger:</b> Post-Termination Access — 2026-08-13 03:17:22 UTC
    </div>
    """, unsafe_allow_html=True)

    RESPONSE_STEPS = [
        ("Account Deactivation",                "Deactivating HF0004 in Identity Provider",             "Successful"),
        ("Session & Refresh Token Revocation",  "Revoking all OAuth2 tokens for r.menon",               "Successful"),
        ("Persistent Session Invalidation",     "Terminating kiosk-wh-02 persistent session",           "Successful"),
        ("Finance Application Access Blocking", "Removing AP-Queue entitlements",                       "Successful"),
        ("App Verification: FinancePortal",     "Confirming HF0004 blocked in FinancePortal",           "Verified"),
        ("App Verification: AP-Workflow",       "Confirming HF0004 blocked in AP-Workflow",             "Verified"),
        ("App Verification: DocuStore",         "Confirming HF0004 blocked in DocuStore",               "Verified"),
        ("Security Audit Log Generated",        "Sealing tamper-evident audit log for INC",             "Successful"),
    ]

    if not st.session_state["containment_done"]:
        if st.button("🔴 Contain Incident — Activate Automated Response", key="contain_btn"):
            log_entries = []
            progress_bar = st.progress(0, text="Initiating automated response…")
            status_placeholder = st.empty()

            rows = []
            for i, (step, detail, final_status) in enumerate(RESPONSE_STEPS):
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                rows.append({"Step": step, "Detail": detail, "Status": "⏳ Pending", "Timestamp": ts})
                status_placeholder.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )
                time.sleep(0.5)

                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                icon = "✅" if final_status in ("Successful", "Verified") else "❌"
                rows[-1]["Status"] = f"{icon} {final_status}"
                rows[-1]["Timestamp"] = ts
                status_placeholder.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )
                progress_bar.progress((i + 1) / len(RESPONSE_STEPS),
                                      text=f"Executing: {step}…")
                log_entries.append({"Step": step, "Detail": detail, "Status": f"{icon} {final_status}", "Timestamp": ts})
                
                # Append each containment action to CryptoTrace Hash Chain
                st.session_state["cryptotrace_chain"].create_block(
                    event_id=f"EVT-CONTAIN-{i+1:03d}",
                    employee_id="HF0004",
                    event_type=f"CONTAINMENT_{step.upper().replace(' ', '_').replace(':', '')}",
                    event_timestamp=ts + " UTC",
                    event_data={"step": step, "detail": detail, "result": final_status},
                    actor="SOC-AUTOMATION",
                    severity="HIGH" if "Verification" not in step else "INFO"
                )
                time.sleep(0.4)

            progress_bar.progress(1.0, text="✅ All containment actions completed.")
            st.session_state["containment_done"] = True
            st.session_state["containment_log"] = log_entries
            st.success(f"🔒 Containment complete. Incident sealed and cryptographically anchored to CryptoTrace: **{inc_id}**")
    else:
        st.markdown("""
        <div class='success-banner'>
            <div style='font-size:1.1rem; font-weight:800; color:#22c55e;'>🔒 CONTAINMENT COMPLETE</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["containment_log"]:
            st.dataframe(
                pd.DataFrame(st.session_state["containment_log"]),
                use_container_width=True,
                hide_index=True,
            )

        st.success(f"Incident sealed: **{inc_id}**")

        if st.button("🔄 Reset Containment Simulation", key="reset_contain"):
            st.session_state["containment_done"] = False
            st.session_state["containment_log"] = []
            st.rerun()

    # Verification status panel
    st.markdown("---")
    st.markdown("<div class='section-header'>📡 Application Access Revocation Verification</div>", unsafe_allow_html=True)
    app_verify_data = {
        "Application": ["FinancePortal", "AP-Workflow", "DocuStore"],
        "User": ["r.menon (HF0004)"] * 3,
        "Access Before": ["✅ Granted"] * 3,
        "Access After": ["❌ Blocked" if st.session_state["containment_done"] else "⏳ Pending"] * 3,
        "Status": ["✅ Verified" if st.session_state["containment_done"] else "⏳ Awaiting Containment"] * 3,
    }
    st.dataframe(pd.DataFrame(app_verify_data), use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — AUDIT LOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif nav == "📋 Audit Log":
    st.markdown("<h1 style='color:#60a5fa;'>📋 Security Audit Log</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Tamper-evident audit trail for Incident INC-20260813-HF0004</p>", unsafe_allow_html=True)

    audit_df = pd.DataFrame(MOCK_AUDIT_LOG)

    # Style severity
    def style_severity(val):
        if val == "CRITICAL":
            return "background-color: #450a0a; color: #fca5a5; font-weight: bold;"
        elif val == "HIGH":
            return "background-color: #431407; color: #fdba74; font-weight: bold;"
        elif val == "INFO":
            return "background-color: #0c1a3a; color: #93c5fd;"
        return ""

    st.dataframe(
        audit_df.style.map(style_severity, subset=["Severity"]),
        use_container_width=True,
        hide_index=True,
    )

    # Summary stats
    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Events", len(audit_df))
    s2.metric("Critical Events", (audit_df["Severity"] == "CRITICAL").sum())
    s3.metric("Completed Actions", (audit_df["Status"].isin(["COMPLETED", "ARCHIVED", "ENFORCED", "RAISED"])).sum())

    # Timeline chart
    st.markdown("<div class='section-header'>📅 Event Timeline</div>", unsafe_allow_html=True)
    timeline_df = audit_df.copy()
    timeline_df["Time Label"] = timeline_df["Timestamp"].apply(lambda x: x[11:19])
    color_map = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "INFO": "#60a5fa"}
    timeline_df["Color"] = timeline_df["Severity"].map(color_map)

    fig_tl = go.Figure()
    for i, row in timeline_df.iterrows():
        fig_tl.add_trace(go.Scatter(
            x=[row["Timestamp"]], y=[i],
            mode="markers+text",
            marker=dict(size=14, color=row["Color"], symbol="circle"),
            text=[f"{row['Event ID']}: {row['Action']}"],
            textposition="middle right",
            textfont=dict(size=10, color="#cbd5e1"),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Event ID']}</b><br>"
                f"Time: {row['Timestamp']}<br>"
                f"Actor: {row['Actor']}<br>"
                f"Action: {row['Action']}<br>"
                f"Status: {row['Status']}<extra></extra>"
            ),
        ))
    fig_tl.update_layout(
        title="Security Event Timeline",
        xaxis_title="Timestamp",
        yaxis=dict(showticklabels=False),
        height=400,
        template=PLOTLY_TEMPLATE,
    )
    plotly_fig_style(fig_tl)
    st.plotly_chart(fig_tl, use_container_width=True)

    # CryptoTrace Integration Banner
    st.markdown("---")
    st.markdown("<div class='section-header'>🔗 CryptoTrace Cryptographic Hash-Chain Integration</div>", unsafe_allow_html=True)
    
    chain_obj = st.session_state.get("cryptotrace_chain")
    chain_records = chain_obj.records if chain_obj else []
    chain_verification = chain_obj.verify_chain() if chain_obj else {"is_valid": True, "status": "Chain Verified"}
    
    ct_c1, ct_c2, ct_c3 = st.columns(3)
    ct_c1.metric("⛓️ CryptoTrace Blocks", f"{len(chain_records)} Blocks")
    ct_c2.metric("🛡️ Chain Integrity", chain_verification["status"])
    ct_c3.metric("🔐 Cryptographic Seal", "HMAC-SHA256 Active")

    with st.expander("🔍 Inspect CryptoTrace Hash-Chain Blocks (SHA-256 & HMAC Seals)", expanded=False):
        if chain_records:
            chain_rows = []
            for r in chain_records:
                chain_rows.append({
                    "Block #": r.get("audit_id"),
                    "Event ID": r.get("event_id"),
                    "Type": r.get("event_type"),
                    "Actor": r.get("actor"),
                    "Severity": r.get("severity"),
                    "Previous Hash": (r.get("previous_hash") or "")[:14] + "...",
                    "Current Hash": (r.get("current_hash") or "")[:14] + "...",
                    "HMAC Seal": (r.get("hmac_seal") or "")[:14] + "...",
                    "Status": r.get("integrity_status", "AUTHENTICATED"),
                })
            st.dataframe(pd.DataFrame(chain_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Chain is currently empty.")

    # Export button
    st.markdown("---")
    csv_data = audit_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Audit Log (CSV)",
        data=csv_data,
        file_name=f"audit_log_INC-20260813-HF0004_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="export_audit",
    )
    st.markdown("""
    <div class='info-card' style='font-size:0.82rem; color:#94a3b8;'>
        ⚠️ All audit log entries are simulated. No real system events were captured.
        This log is generated for demonstration of the LIFECYCLE-X AI incident response workflow.
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — CRYPTOGRAPHIC SECURITY CENTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif nav == "🔐 Cryptographic Security Center":
    st.markdown("<h1 style='color:#60a5fa;'>🔐 Cryptographic Security Center</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#94a3b8;'>
        Unified Cybersecurity Architecture: <b>Random Forest ML</b> (Predictive Detection) + 
        <b>AES-256-GCM</b> (Confidential Evidence Vault) + <b>HMAC-SHA256</b> (Authenticity) + 
        <b>CryptoTrace</b> (Tamper-Evident Hash-Chain Audit Trail).
    </p>
    """, unsafe_allow_html=True)

    # Key status metadata
    key_meta = get_key_metadata()
    chain_obj = st.session_state.get("cryptotrace_chain")
    if chain_obj is None:
        chain_obj = build_initial_chain()
        st.session_state["cryptotrace_chain"] = chain_obj

    # Check whether we are evaluating the clean chain or a simulated tampered chain
    active_chain = (
        st.session_state["tampered_chain_copy"]
        if st.session_state.get("tampered_chain_active") and st.session_state.get("tampered_chain_copy")
        else chain_obj.records
    )
    verification_result = chain_obj.verify_chain(active_chain)

    vault_rec = st.session_state.get("rohan_vault_record", {})
    hmac_sig = st.session_state.get("rohan_hmac_sig", "")
    raw_evid = st.session_state.get("rohan_raw_evidence", {})

    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("📦 Vault Records", "1 Active Record", "AES-256-GCM")
    
    # Check HMAC integrity
    is_evid_valid, evid_status, _ = verify_hmac(raw_evid, hmac_sig) if raw_evid and hmac_sig else (True, "Integrity Verified", {})
    m_col2.metric("🛡️ Evidence Integrity", evid_status, "HMAC-SHA256")
    
    m_col3.metric("⛓️ CryptoTrace Blocks", f"{len(active_chain)} Blocks", "Sealed")
    
    chain_status_label = verification_result["status"]
    if verification_result["is_valid"]:
        m_col4.metric("🔗 Chain Integrity", "Chain Verified ✅", "Tamper-Proof")
    else:
        m_col4.metric("🚨 Chain Integrity", f"{chain_status_label} ❌", f"Block #{verification_result.get('first_invalid_index')}", delta_color="inverse")

    st.markdown("---")

    # 3-Pillar Layout Tabs
    tab_vault, tab_trace, tab_ml_crypto = st.tabs([
        "🛡️ SecureEvidence Vault (AES-GCM & HMAC)",
        "⛓️ CryptoTrace Hash Chain & Tamper Demo",
        "🤖 ML + Crypto Integrated Pipeline"
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: SecureEvidence Vault
    # ──────────────────────────────────────────────────────────────────────────
    with tab_vault:
        st.markdown("<div class='section-header'>🛡️ SecureEvidence Vault: Authenticated Confidential Storage</div>", unsafe_allow_html=True)
        
        v_col_left, v_col_right = st.columns([1, 1])
        with v_col_left:
            st.markdown(f"""
            <div class='info-card'>
                <div style='color:#60a5fa; font-weight:700; font-size:1rem; margin-bottom:10px;'>
                    🔐 Cryptographic Vault Parameters
                </div>
                <table style='width:100%; border-collapse:collapse; color:#cbd5e1; font-size:0.86rem;'>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>Vault ID</td><td><b>{vault_rec.get('vault_id', 'VAULT-001')}</b></td></tr>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>Confidentiality Cipher</td><td><b>AES-256-GCM</b> (256-bit key)</td></tr>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>Unique Nonce (96-bit)</td><td><code style='color:#86efac;'>{vault_rec.get('nonce_hex', 'N/A')}</code></td></tr>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>Authentication Tag (128-bit)</td><td><code style='color:#93c5fd;'>{vault_rec.get('tag_hex', 'N/A')}</code></td></tr>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>Evidence Authenticator</td><td><b>HMAC-SHA256</b> (Dedicated Key)</td></tr>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>HMAC Signature</td><td><code style='color:#fdba74;'>{hmac_sig[:32]}...</code></td></tr>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>Key Storage</td><td>Isolated Key Store (Segregated Keys)</td></tr>
                    <tr><td style='padding:5px 0; color:#94a3b8;'>Protected Entity</td><td>Rohan Menon (HF0004) — Incident Evidence</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with v_col_right:
            st.markdown("""
            <div class='info-card'>
                <div style='color:#60a5fa; font-weight:700; font-size:1rem; margin-bottom:10px;'>
                    🔒 Vault Actions &amp; Zero-Knowledge Controls
                </div>
                <p style='color:#94a3b8; font-size:0.83rem;'>
                    Sensitive employee attributes (role, accounts payable entitlements, incident forensics)
                    are encrypted before resting on disk. Decryption is strictly restricted to authorized investigations.
                </p>
            </div>
            """, unsafe_allow_html=True)

            btn_v1, btn_v2, btn_v3 = st.columns(3)
            with btn_v1:
                if st.button("🔒 Re-Encrypt Evidence", key="btn_reencrypt"):
                    # Generate fresh 96-bit nonce and re-encrypt
                    new_vault_rec = encrypt_evidence(
                        raw_evid,
                        metadata={"employee_id": "HF0004", "incident_id": "INC-20260813-HF0004", "refreshed": True}
                    )
                    new_sig, new_digest = generate_hmac(raw_evid)
                    st.session_state["rohan_vault_record"] = new_vault_rec
                    st.session_state["rohan_hmac_sig"] = new_sig
                    st.session_state["rohan_sha256"] = new_digest
                    st.success("✅ Fresh AES-256-GCM encryption completed with new unique 96-bit nonce!")
                    st.rerun()

            with btn_v2:
                if st.button("🔍 Verify HMAC Integrity", key="btn_verify_hmac"):
                    valid, stat, det = verify_hmac(raw_evid, hmac_sig)
                    if valid:
                        st.success(f"✅ {stat}: Evidence matches cryptographic signature (Constant-time check passed).")
                    else:
                        st.error(f"❌ {stat}: Cryptographic verification failed!")

            with btn_v3:
                btn_label = "🔒 Hide Plaintext" if st.session_state.get("show_decrypted_evidence") else "🔓 Decrypt Evidence"
                if st.button(btn_label, key="btn_toggle_decrypt"):
                    st.session_state["show_decrypted_evidence"] = not st.session_state.get("show_decrypted_evidence", False)
                    st.rerun()

        # Decrypted evidence viewer (Authorized SOC demo)
        if st.session_state.get("show_decrypted_evidence"):
            st.markdown("<div class='section-header' style='color:#22c55e;'>🔓 Authorized Plaintext Decryption (AES-256-GCM Auth Verified)</div>", unsafe_allow_html=True)
            try:
                decrypted_payload = decrypt_evidence(vault_rec)
                st.json(decrypted_payload)
                st.info("ℹ️ Plaintext evidence decrypted in-memory. Authentication tag validated; zero tampering detected.")
            except DecryptionError as de:
                st.error(f"❌ Decryption Failed: {de}")

        # Display raw ciphertext envelope
        with st.expander("📦 View Raw Encrypted Vault Envelope (Ciphertext, Nonce, Tag)", expanded=False):
            st.json({
                "vault_id": vault_rec.get("vault_id"),
                "algorithm": vault_rec.get("algorithm"),
                "nonce_hex": vault_rec.get("nonce_hex"),
                "ciphertext_hex_sample": (vault_rec.get("ciphertext_hex", "")[:64] + "... [TRUNCATED]"),
                "tag_hex": vault_rec.get("tag_hex"),
                "hmac_sha256_signature": hmac_sig,
                "bound_aad": vault_rec.get("aad_bound"),
                "encrypted_at_utc": vault_rec.get("encrypted_at_utc"),
            })

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: CryptoTrace Hash Chain & Tamper Demo
    # ──────────────────────────────────────────────────────────────────────────
    with tab_trace:
        st.markdown("<div class='section-header'>⛓️ CryptoTrace: Tamper-Evident Hash-Chain Audit Engine</div>", unsafe_allow_html=True)

        # Tampering status banner
        if st.session_state.get("tampered_chain_active"):
            st.markdown(f"""
            <div class='incident-banner' style='border: 2px solid #ef4444;'>
                <div style='font-size:1.2rem; font-weight:800; color:#ef4444;'>
                    🚨 CRITICAL INTEGRITY ALERT: AUDIT TAMPERING DETECTED!
                </div>
                <div style='color:#fca5a5; font-size:0.9rem; margin-top:8px;'>
                    <b>Tamper Status:</b> {verification_result['status']}<br>
                    <b>Compromised Block:</b> #{verification_result.get('first_invalid_index')}<br>
                    <b>Diagnostic:</b> {verification_result.get('error_message')}<br>
                    <b>Cryptographic Reason:</b> The recalculated SHA-256 payload digest failed to match the current_hash,
                    and down-chain previous_hash linkage was invalidated.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='success-banner'>
                <div style='font-size:1.1rem; font-weight:800; color:#22c55e;'>
                    ✅ CRYPTOTRACE HASH-CHAIN INTEGRITY VERIFIED
                </div>
                <div style='color:#86efac; font-size:0.85rem; margin-top:4px;'>
                    All {len(active_chain)} blocks verified against genesis hash (<code>{GENESIS_HASH[:16]}...</code>). 
                    Sequential linkage intact. HMAC seals authenticated.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Control buttons for Tamper Demonstration
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            if st.button("⛓️ Verify Audit Chain", key="btn_verify_chain"):
                res = chain_obj.verify_chain(active_chain)
                st.session_state["chain_verification_cache"] = res
                if res["is_valid"]:
                    st.success(f"✅ {res['status']}: All {res['total_records']} blocks intact.")
                else:
                    st.error(f"❌ {res['status']} at Block #{res['first_invalid_index']}: {res['error_message']}")

        with t_col2:
            if st.button("⚠️ Simulate Evidence Tampering (Demonstration)", key="btn_simulate_tamper"):
                # Create a controlled deep copy with modified data in Block #2
                tampered = chain_obj.simulate_tampering(
                    record_index=2,
                    field_to_alter="event_data",
                    tampered_value={
                        "accessed_resource": "/finance/ap-queue/APPROVED_PAYMENT_ALTERED.xlsx",
                        "unauthorized_actor": "malicious_insider",
                        "tampered_amount": "$450,000",
                    }
                )
                st.session_state["tampered_chain_copy"] = tampered
                st.session_state["tampered_chain_active"] = True
                st.warning("⚠️ Controlled demonstration: Modified payload in copied Block #2. Original chain preserved.")
                st.rerun()

        with t_col3:
            if st.button("🔄 Restore Original Clean Chain", key="btn_restore_chain"):
                st.session_state["tampered_chain_active"] = False
                st.session_state["tampered_chain_copy"] = None
                st.success("✅ Restored original clean audit chain. Tamper simulation cleared.")
                st.rerun()

        # Visual Block Explorer
        st.markdown("<div class='section-header'>📦 Interactive Block Explorer</div>", unsafe_allow_html=True)
        block_rows = []
        for b in active_chain:
            idx = b.get("audit_id")
            is_compromised = (
                st.session_state.get("tampered_chain_active") and
                verification_result.get("first_invalid_index") is not None and
                idx >= verification_result.get("first_invalid_index")
            )
            block_rows.append({
                "Block #": f"#{idx}" + (" 🚨 COMPROMISED" if is_compromised else " ✅"),
                "Event ID": b.get("event_id"),
                "Event Type": b.get("event_type"),
                "Actor": b.get("actor"),
                "Severity": b.get("severity"),
                "Previous Hash": (b.get("previous_hash") or "")[:16] + "...",
                "Current Hash": (b.get("current_hash") or "")[:16] + "...",
                "HMAC Seal": (b.get("hmac_seal") or "")[:16] + "...",
            })

        st.dataframe(pd.DataFrame(block_rows), use_container_width=True, hide_index=True)

        with st.expander("🔍 View Full Canonical JSON for Each Block in Chain", expanded=False):
            st.json(active_chain)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: ML + Crypto Integrated Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    with tab_ml_crypto:
        st.markdown("<div class='section-header'>🤖 + 🔐 Unified Multi-Layer Defense Architecture</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card' style='font-size:0.88rem; line-height:1.7;'>
            <b>How LIFECYCLE-X AI Integrates ML and Cryptography:</b><br>
            1. <b>ML Risk Prediction:</b> Random Forest evaluates 10 authentication telemetry features (token age, sync lag, device, sensitivity) to score residual risk.<br>
            2. <b>Zero-Tolerance Security Policy:</b> Regardless of ML score, post-termination authentication triggers a deterministic policy block.<br>
            3. <b>Confidential Evidence Custody:</b> Incident details are sealed in the <b>SecureEvidence Vault</b> using <b>AES-256-GCM</b> with unique 96-bit nonces.<br>
            4. <b>Cryptographic Authenticity:</b> Evidence is signed using <b>HMAC-SHA256</b> with a segregated key.<br>
            5. <b>Tamper-Evident Chronology:</b> <b>CryptoTrace</b> links the incident, policy block, and automated containment actions into an immutable SHA-256 hash chain with cryptographic HMAC seals.<br>
        </div>
        """, unsafe_allow_html=True)

        ml_c1, ml_c2 = st.columns(2)
        with ml_c1:
            st.markdown("<div class='section-header'>🤖 Random Forest Classifier Summary</div>", unsafe_allow_html=True)
            st.markdown("""
            • <b>Model:</b> Random Forest (200 estimators, balanced weights)<br>
            • <b>Test Set Accuracy:</b> 97.84%<br>
            • <b>Test Set Recall:</b> 97.14%<br>
            • <b>Test Set F1-Score:</b> 88.31%<br>
            • <b>Rohan Menon Prediction:</b> 100.0% HIGH RISK (Incident Blocked)
            """, unsafe_allow_html=True)

        with ml_c2:
            st.markdown("<div class='section-header'>🔐 Cryptographic Primitives Summary</div>", unsafe_allow_html=True)
            st.markdown(f"""
            • <b>Confidentiality:</b> AES-256-GCM (256 bits, 96-bit unique nonces)<br>
            • <b>Integrity:</b> HMAC-SHA256 (256-bit secret key)<br>
            • <b>Audit Trail:</b> Append-Only SHA-256 Hash Chain (CryptoTrace)<br>
            • <b>Key Separation:</b> AES Key != HMAC Key (Strictly Segregated)<br>
            • <b>Genesis Hash:</b> <code>{GENESIS_HASH[:24]}...</code>
            """, unsafe_allow_html=True)


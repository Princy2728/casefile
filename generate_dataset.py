"""
LIFECYCLE-X AI — Synthetic Dataset Generator
Generates ~2000 employee authentication records for training.
All data is completely synthetic and for demonstration purposes only.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproducibility
RNG = np.random.default_rng(42)
np.random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "auth_logs.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

NUM_RECORDS = 2000
NUM_EMPLOYEES = 100

# ------------------------------------------------------------------
# 1. Build employee master table
# ------------------------------------------------------------------
employee_ids = [f"HF{str(i).zfill(4)}" for i in range(1, NUM_EMPLOYEES + 1)]

# ~30% terminated
terminated_count = 30
active_count = NUM_EMPLOYEES - terminated_count

statuses = ["terminated"] * terminated_count + ["active"] * active_count
np.random.shuffle(statuses)

# Termination dates for terminated employees (past 2 years)
def random_past_date(start_days=30, end_days=730):
    base = datetime(2026, 11, 1)
    delta = np.random.randint(start_days, end_days)
    return (base - timedelta(days=delta)).strftime("%Y-%m-%d")

termination_dates = {}
for eid, status in zip(employee_ids, statuses):
    if status == "terminated":
        termination_dates[eid] = random_past_date()
    else:
        termination_dates[eid] = None  # active employees have no termination date

# ------------------------------------------------------------------
# 2. Force HF0004 (Rohan Menon) profile
# ------------------------------------------------------------------
termination_dates["HF0004"] = "2026-06-30"
# Find HF0004 index and ensure terminated
idx4 = employee_ids.index("HF0004")
statuses[idx4] = "terminated"

# ------------------------------------------------------------------
# 3. Generate authentication records
# ------------------------------------------------------------------
records = []

session_types = ["persistent", "standard"]
device_categories = ["corporate", "kiosk", "personal", "unknown"]
mfa_statuses = ["evaluated", "not_evaluated", "bypassed"]
sensitivity_levels = ["low", "medium", "high", "critical"]

def compute_risk(row):
    """Deterministic risk label — no target leakage from raw columns."""
    score = 0
    # Post-termination access
    if row["employment_status"] == "terminated":
        if row["auth_success"] == 1:
            if row["termination_date"] is not None:
                try:
                    term_dt = datetime.strptime(row["termination_date"], "%Y-%m-%d")
                    auth_dt = datetime.strptime(row["auth_timestamp"][:19], "%Y-%m-%d %H:%M:%S")
                    if auth_dt > term_dt:
                        score += 4  # strong signal
                except Exception:
                    pass
    # Persistent session
    if row["session_type"] == "persistent":
        score += 1
    # MFA not evaluated or bypassed
    if row["mfa_status"] in ("not_evaluated", "bypassed"):
        score += 1
    # High/Critical application
    if row["application_sensitivity"] in ("high", "critical"):
        score += 1
    # Non-corporate device
    if row["device_category"] in ("kiosk", "personal", "unknown"):
        score += 1
    # Long sync delay
    if row["hr_idp_sync_delay_hours"] > 24:
        score += 1
    # Revocation not acknowledged
    if row["revocation_acknowledged"] == 0:
        score += 1
    return 1 if score >= 5 else 0

for i in range(NUM_RECORDS):
    eid = np.random.choice(employee_ids)
    status = statuses[employee_ids.index(eid)]
    term_date = termination_dates[eid]

    # Auth timestamp — recent 18 months
    base_ts = datetime(2026, 11, 1)
    delta_days = np.random.randint(1, 548)
    auth_ts = base_ts - timedelta(days=delta_days, hours=np.random.randint(0, 24),
                                  minutes=np.random.randint(0, 60))

    # Auth success: terminated employees have ~20% chance of success (residual access)
    if status == "terminated":
        auth_success = np.random.choice([0, 1], p=[0.80, 0.20])
    else:
        auth_success = np.random.choice([0, 1], p=[0.05, 0.95])

    session_type = np.random.choice(session_types, p=[0.3, 0.7])
    device_cat = np.random.choice(device_categories, p=[0.55, 0.20, 0.15, 0.10])
    mfa_stat = np.random.choice(mfa_statuses, p=[0.65, 0.25, 0.10])
    token_age = round(np.random.exponential(scale=48), 2)  # hours
    sync_delay = round(np.random.exponential(scale=12), 2)  # hours
    app_sens = np.random.choice(sensitivity_levels, p=[0.30, 0.35, 0.25, 0.10])
    revoc_ack = np.random.choice([0, 1], p=[0.3, 0.7])

    row = {
        "employee_id": eid,
        "employment_status": status,
        "termination_date": term_date,
        "auth_timestamp": auth_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "session_type": session_type,
        "token_age_hours": token_age,
        "device_category": device_cat,
        "mfa_status": mfa_stat,
        "hr_idp_sync_delay_hours": sync_delay,
        "application_sensitivity": app_sens,
        "revocation_acknowledged": revoc_ack,
        "auth_success": auth_success,
        "risk_label": 0,  # placeholder, computed below
    }
    row["risk_label"] = compute_risk(row)
    records.append(row)

# ------------------------------------------------------------------
# 4. Force Rohan Menon record (index 0 for easy reference)
# ------------------------------------------------------------------
rohan_record = {
    "employee_id": "HF0004",
    "employment_status": "terminated",
    "termination_date": "2026-06-30",
    "auth_timestamp": "2026-08-13 03:17:22",
    "session_type": "persistent",
    "token_age_hours": 312.5,
    "device_category": "kiosk",
    "mfa_status": "not_evaluated",
    "hr_idp_sync_delay_hours": 44.0,
    "application_sensitivity": "critical",
    "revocation_acknowledged": 0,
    "auth_success": 1,
    "risk_label": 1,
}
records.insert(0, rohan_record)

# ------------------------------------------------------------------
# 5. Add more post-termination high-risk records for realism
#    Each terminated employee can have multiple attack attempts
# ------------------------------------------------------------------
extra_risk = []
term_eids = [e for e, s in zip(employee_ids, statuses) if s == "terminated" and e != "HF0004"]

# Pick 25 terminated employees, give each 2–4 post-termination access attempts
selected_attackers = np.random.choice(term_eids, size=min(25, len(term_eids)), replace=False)
for eid in selected_attackers:
    term_d = termination_dates[eid]
    term_dt = datetime.strptime(term_d, "%Y-%m-%d")
    num_attempts = np.random.randint(2, 5)  # 2 to 4 attacks per employee
    for attempt in range(num_attempts):
        auth_ts = term_dt + timedelta(days=np.random.randint(1, 120),
                                      hours=np.random.randint(0, 24),
                                      minutes=np.random.randint(0, 60))
        row = {
            "employee_id": eid,
            "employment_status": "terminated",
            "termination_date": term_d,
            "auth_timestamp": auth_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "session_type": np.random.choice(["persistent", "standard"], p=[0.7, 0.3]),
            "token_age_hours": round(np.random.uniform(48, 600), 2),
            "device_category": np.random.choice(["kiosk", "personal", "unknown"], p=[0.5, 0.3, 0.2]),
            "mfa_status": np.random.choice(["not_evaluated", "bypassed"], p=[0.6, 0.4]),
            "hr_idp_sync_delay_hours": round(np.random.uniform(12, 96), 2),
            "application_sensitivity": np.random.choice(["high", "critical"], p=[0.5, 0.5]),
            "revocation_acknowledged": 0,
            "auth_success": 1,
            "risk_label": 1,
        }
        extra_risk.append(row)

# Also add Rohan Menon with 3 more post-termination access attempts
rohan_extra_attacks = [
    {
        "employee_id": "HF0004", "employment_status": "terminated",
        "termination_date": "2026-06-30",
        "auth_timestamp": "2026-07-14 22:03:11", "session_type": "persistent",
        "token_age_hours": 168.0, "device_category": "kiosk",
        "mfa_status": "not_evaluated", "hr_idp_sync_delay_hours": 44.0,
        "application_sensitivity": "high", "revocation_acknowledged": 0,
        "auth_success": 1, "risk_label": 1,
    },
    {
        "employee_id": "HF0004", "employment_status": "terminated",
        "termination_date": "2026-06-30",
        "auth_timestamp": "2026-07-29 01:44:38", "session_type": "persistent",
        "token_age_hours": 240.0, "device_category": "kiosk",
        "mfa_status": "bypassed", "hr_idp_sync_delay_hours": 44.0,
        "application_sensitivity": "critical", "revocation_acknowledged": 0,
        "auth_success": 1, "risk_label": 1,
    },
    {
        "employee_id": "HF0004", "employment_status": "terminated",
        "termination_date": "2026-06-30",
        "auth_timestamp": "2026-09-05 04:22:55", "session_type": "persistent",
        "token_age_hours": 410.0, "device_category": "personal",
        "mfa_status": "not_evaluated", "hr_idp_sync_delay_hours": 44.0,
        "application_sensitivity": "critical", "revocation_acknowledged": 0,
        "auth_success": 1, "risk_label": 1,
    },
]
extra_risk.extend(rohan_extra_attacks)

records.extend(extra_risk)

# ------------------------------------------------------------------
# 6. Build DataFrame and save
# ------------------------------------------------------------------
df = pd.DataFrame(records)

# Separate the forced Rohan Menon incident record (first inserted record)
# The forced record has auth_timestamp of "2026-08-13 03:17:22"
rohan_forced = df[df["auth_timestamp"] == "2026-08-13 03:17:22"].head(1)
rest_df = df[df["auth_timestamp"] != "2026-08-13 03:17:22"]

# Shuffle the rest
rest_df = rest_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Put Rohan forced record at index 0
df = pd.concat([rohan_forced, rest_df], ignore_index=True)

df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset saved to: {OUTPUT_PATH}")
print(f"   Total records   : {len(df)}")
print(f"   Risk label=1    : {df['risk_label'].sum()} ({df['risk_label'].mean()*100:.1f}%)")
print(f"   Risk label=0    : {(df['risk_label']==0).sum()}")
print(f"\nFirst 3 records:\n{df.head(3).to_string()}")

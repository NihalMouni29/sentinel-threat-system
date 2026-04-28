"""
AI Insider Threat Detection Engine
Uses Isolation Forest + behavioral profiling for anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# SYNTHETIC LOG GENERATOR
# ─────────────────────────────────────────────

USERS = {
    "alice.morgan": {"dept": "Finance", "role": "Analyst", "normal_hour_start": 8, "normal_hour_end": 17, "avg_files": 12},
    "bob.chen": {"dept": "Engineering", "role": "Developer", "normal_hour_start": 9, "normal_hour_end": 18, "avg_files": 35},
    "carol.davis": {"dept": "HR", "role": "Manager", "normal_hour_start": 8, "normal_hour_end": 16, "avg_files": 8},
    "dave.wilson": {"dept": "Sales", "role": "Rep", "normal_hour_start": 7, "normal_hour_end": 19, "avg_files": 20},
    "eve.johnson": {"dept": "IT", "role": "SysAdmin", "normal_hour_start": 6, "normal_hour_end": 22, "avg_files": 50},
}

FILE_CATEGORIES = ["contracts", "employee_records", "financial_reports", "source_code", "customer_data", "internal_memos", "confidential"]
ACTIONS = ["read", "download", "copy", "delete", "upload", "share", "print"]
LOCATIONS = ["office", "office", "office", "vpn_remote", "vpn_remote", "unknown_ip", "tor_exit_node"]


def generate_normal_log(user_id, base_date):
    profile = USERS[user_id]
    hour = random.randint(profile["normal_hour_start"], profile["normal_hour_end"])
    minute = random.randint(0, 59)
    dt = base_date.replace(hour=hour, minute=minute)
    files_accessed = max(1, int(np.random.normal(profile["avg_files"], profile["avg_files"] * 0.3)))
    
    return {
        "timestamp": dt.isoformat(),
        "user_id": user_id,
        "hour": hour,
        "day_of_week": dt.weekday(),
        "files_accessed": files_accessed,
        "data_volume_mb": round(files_accessed * random.uniform(0.5, 3.0), 2),
        "failed_logins": random.choices([0, 1], weights=[95, 5])[0],
        "unique_file_categories": random.randint(1, 3),
        "sensitive_files": random.randint(0, max(1, files_accessed // 5)),
        "actions_delete": random.choices([0, 1], weights=[90, 10])[0],
        "location_risk": 0 if random.random() > 0.1 else 1,
        "typing_speed_wpm": random.randint(45, 85),
        "session_duration_min": random.randint(30, 480),
        "after_hours": 0,
        "weekend": 1 if dt.weekday() >= 5 else 0,
        "is_anomaly": False,
        "anomaly_type": "normal",
        "dept": profile["dept"],
        "role": profile["role"],
    }


def generate_anomalous_log(user_id, base_date, anomaly_type):
    log = generate_normal_log(user_id, base_date)
    
    if anomaly_type == "data_exfiltration":
        log["hour"] = random.choice([0, 1, 2, 3, 22, 23])
        log["files_accessed"] = random.randint(200, 600)
        log["data_volume_mb"] = round(log["files_accessed"] * random.uniform(5, 15), 2)
        log["sensitive_files"] = log["files_accessed"] // 2
        log["after_hours"] = 1
        log["location_risk"] = random.choice([1, 2])
        log["unique_file_categories"] = random.randint(4, 7)
        
    elif anomaly_type == "credential_abuse":
        log["failed_logins"] = random.randint(5, 20)
        log["location_risk"] = 2
        log["files_accessed"] = random.randint(1, 5)
        log["hour"] = random.choice([0, 1, 2, 14, 15])
        
    elif anomaly_type == "sabotage":
        log["actions_delete"] = random.randint(10, 50)
        log["files_accessed"] = random.randint(20, 80)
        log["hour"] = random.choice([22, 23, 0, 1])
        log["after_hours"] = 1
        log["data_volume_mb"] = 0.1
        
    elif anomaly_type == "policy_violation":
        log["weekend"] = 1
        log["after_hours"] = 1
        log["location_risk"] = 1
        log["sensitive_files"] = random.randint(15, 40)
        log["unique_file_categories"] = 5
        log["data_volume_mb"] = random.uniform(200, 800)

    log["timestamp"] = base_date.replace(hour=log["hour"], minute=random.randint(0,59)).isoformat()
    log["is_anomaly"] = True
    log["anomaly_type"] = anomaly_type
    return log


def generate_dataset(days=90):
    logs = []
    base = datetime(2025, 1, 1)
    
    for day_offset in range(days):
        current_day = base + timedelta(days=day_offset)
        if current_day.weekday() >= 5:
            continue
            
        for user_id in USERS:
            # Normal sessions
            for _ in range(random.randint(1, 3)):
                logs.append(generate_normal_log(user_id, current_day))
            
            # Inject anomalies (5% chance per day per user)
            if random.random() < 0.05:
                anomaly = random.choice(["data_exfiltration", "credential_abuse", "sabotage", "policy_violation"])
                logs.append(generate_anomalous_log(user_id, current_day, anomaly))
    
    df = pd.DataFrame(logs)
    df.to_csv("data/behavior_logs.csv", index=False)
    print(f"✅ Generated {len(df)} log entries ({df['is_anomaly'].sum()} anomalies)")
    return df


# ─────────────────────────────────────────────
# ISOLATION FOREST MODEL
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "hour", "day_of_week", "files_accessed", "data_volume_mb",
    "failed_logins", "unique_file_categories", "sensitive_files",
    "actions_delete", "location_risk", "typing_speed_wpm",
    "session_duration_min", "after_hours", "weekend"
]


class ThreatDetectionEngine:
    def __init__(self):
        self.models = {}          # per-user models
        self.scalers = {}
        self.global_model = None
        self.global_scaler = None
        self.df = None
        self.results = None

    def train(self, df):
        self.df = df.copy()
        
        # Global model
        X = df[FEATURE_COLS].fillna(0)
        self.global_scaler = StandardScaler()
        X_scaled = self.global_scaler.fit_transform(X)
        self.global_model = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=42, n_jobs=-1
        )
        self.global_model.fit(X_scaled)
        
        # Per-user models
        for user_id in df["user_id"].unique():
            user_df = df[df["user_id"] == user_id][FEATURE_COLS].fillna(0)
            if len(user_df) < 5:
                continue
            scaler = StandardScaler()
            X_u = scaler.fit_transform(user_df)
            model = IsolationForest(
                n_estimators=100, contamination=0.05,
                random_state=42
            )
            model.fit(X_u)
            self.models[user_id] = model
            self.scalers[user_id] = scaler
        
        print("✅ Models trained")

    def predict(self, df):
        results = df.copy()
        X = df[FEATURE_COLS].fillna(0)
        
        # Global scores
        X_scaled = self.global_scaler.transform(X)
        global_scores = self.global_model.decision_function(X_scaled)
        global_pred = self.global_model.predict(X_scaled)
        
        results["global_anomaly_score"] = -global_scores  # higher = more anomalous
        results["global_flagged"] = (global_pred == -1)
        
        # Per-user scores
        user_scores = []
        for _, row in df.iterrows():
            uid = row["user_id"]
            if uid in self.models:
                x = np.array(row[FEATURE_COLS].fillna(0)).reshape(1, -1)
                x_s = self.scalers[uid].transform(x)
                score = -self.models[uid].decision_function(x_s)[0]
            else:
                score = results.loc[_, "global_anomaly_score"] if _ in results.index else 0.5
            user_scores.append(score)
        
        results["user_anomaly_score"] = user_scores
        
        # Combined risk score (0-100)
        results["risk_score"] = (
            0.5 * results["global_anomaly_score"] + 
            0.5 * results["user_anomaly_score"]
        )
        # Normalize to 0-100
        min_r, max_r = results["risk_score"].min(), results["risk_score"].max()
        results["risk_score"] = ((results["risk_score"] - min_r) / (max_r - min_r + 1e-9) * 100).round(1)
        
        # Intent classification
        results["intent"] = results.apply(self._classify_intent, axis=1)
        results["flagged"] = results["risk_score"] > 65
        
        self.results = results
        return results

    def _classify_intent(self, row):
        if row["failed_logins"] > 3 and row["location_risk"] > 0:
            return "malicious"
        elif row["files_accessed"] > 100 or row["data_volume_mb"] > 500:
            return "malicious"
        elif row["after_hours"] and row["sensitive_files"] > 5:
            return "suspicious"
        elif row["weekend"] and row["data_volume_mb"] > 50:
            return "careless"
        elif row["unique_file_categories"] > 4:
            return "curious"
        else:
            return "normal"

    def get_user_timeline(self, user_id):
        if self.results is None:
            return []
        user_data = self.results[self.results["user_id"] == user_id].copy()
        user_data = user_data.sort_values("timestamp")
        return user_data.to_dict("records")

    def get_alerts(self):
        if self.results is None:
            return []
        flagged = self.results[self.results["flagged"]].copy()
        flagged = flagged.sort_values("risk_score", ascending=False)
        return flagged.to_dict("records")

    def predict_next_action(self, user_id):
        if self.results is None:
            return {}
        user_data = self.results[self.results["user_id"] == user_id]
        if user_data.empty:
            return {}
        
        last = user_data.sort_values("timestamp").iloc[-1]
        intent = last["intent"]
        risk = last["risk_score"]
        
        predictions = {
            "malicious": {
                "next_action": "Mass file download or deletion attempt",
                "probability": round(min(risk / 100 + 0.2, 0.95), 2),
                "timeframe": "Next 2-6 hours",
                "recommended_action": "🔴 IMMEDIATE: Suspend account & alert security team"
            },
            "suspicious": {
                "next_action": "Escalated data access or lateral movement",
                "probability": round(risk / 100, 2),
                "timeframe": "Next 12-24 hours",
                "recommended_action": "🟠 Monitor closely, prepare for account lock"
            },
            "curious": {
                "next_action": "Exploring sensitive directories outside job scope",
                "probability": round(risk / 100 * 0.7, 2),
                "timeframe": "Next 1-3 days",
                "recommended_action": "🟡 Send policy reminder, increase monitoring"
            },
            "careless": {
                "next_action": "Accidental data exposure via wrong sharing settings",
                "probability": round(risk / 100 * 0.5, 2),
                "timeframe": "Next 1-5 days",
                "recommended_action": "🟡 Security awareness training required"
            },
            "normal": {
                "next_action": "Continued normal work activity",
                "probability": 0.92,
                "timeframe": "Ongoing",
                "recommended_action": "✅ No action required"
            }
        }
        return predictions.get(intent, predictions["normal"])

    def get_dashboard_stats(self):
        if self.results is None:
            return {}
        r = self.results
        return {
            "total_events": len(r),
            "flagged_events": int(r["flagged"].sum()),
            "active_users": int(r["user_id"].nunique()),
            "high_risk_users": int((r.groupby("user_id")["risk_score"].max() > 65).sum()),
            "avg_risk_score": round(r["risk_score"].mean(), 1),
            "intent_breakdown": r["intent"].value_counts().to_dict(),
            "anomaly_types": {
                "data_exfiltration": int(r["anomaly_type"].eq("data_exfiltration").sum()),
                "credential_abuse": int(r["anomaly_type"].eq("credential_abuse").sum()),
                "sabotage": int(r["anomaly_type"].eq("sabotage").sum()),
                "policy_violation": int(r["anomaly_type"].eq("policy_violation").sum()),
            }
        }


# ─────────────────────────────────────────────
# SINGLETON ENGINE INSTANCE
# ─────────────────────────────────────────────

engine = ThreatDetectionEngine()


def initialize_engine():
    import os
    os.makedirs("data", exist_ok=True)
    
    if not pd.io.common.file_exists("data/behavior_logs.csv"):
        df = generate_dataset(days=90)
    else:
        df = pd.read_csv("data/behavior_logs.csv")
        print(f"✅ Loaded {len(df)} existing log entries")
    
    engine.train(df)
    engine.predict(df)
    print("✅ Engine initialized and ready")
    return engine

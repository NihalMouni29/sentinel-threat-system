"""
Flask API for AI Insider Threat Detection System
"""

from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
from datetime import datetime
import json
import threading
import time

from threat_engine import initialize_engine, generate_anomalous_log, generate_normal_log, USERS
from datetime import datetime

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Initialize engine on startup
print("🚀 Initializing Threat Detection Engine...")
engine = initialize_engine()
print("🟢 System ONLINE")


def json_serialize(obj):
    """Handle numpy/pandas types for JSON"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def get_stats():
    stats = engine.get_dashboard_stats()
    return jsonify(stats)


@app.route("/api/alerts")
def get_alerts():
    alerts = engine.get_alerts()
    # Return top 20 alerts
    serializable = []
    for a in alerts[:20]:
        clean = {}
        for k, v in a.items():
            try:
                clean[k] = json_serialize(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            except:
                clean[k] = str(v)
        serializable.append(clean)
    return jsonify(serializable)


@app.route("/api/users")
def get_users():
    if engine.results is None:
        return jsonify([])
    
    user_summaries = []
    for user_id, profile in USERS.items():
        user_data = engine.results[engine.results["user_id"] == user_id]
        if user_data.empty:
            continue
        
        max_risk = float(user_data["risk_score"].max())
        avg_risk = float(user_data["risk_score"].mean())
        last_intent = user_data.sort_values("timestamp").iloc[-1]["intent"]
        flagged_count = int(user_data["flagged"].sum())
        last_seen = user_data["timestamp"].max()
        
        user_summaries.append({
            "user_id": user_id,
            "name": user_id.replace(".", " ").title(),
            "dept": profile["dept"],
            "role": profile["role"],
            "max_risk_score": round(max_risk, 1),
            "avg_risk_score": round(avg_risk, 1),
            "intent": last_intent,
            "flagged_events": flagged_count,
            "last_seen": last_seen,
            "status": "critical" if max_risk > 80 else ("warning" if max_risk > 60 else "normal")
        })
    
    user_summaries.sort(key=lambda x: x["max_risk_score"], reverse=True)
    return jsonify(user_summaries)


@app.route("/api/user/<user_id>/timeline")
def get_user_timeline(user_id):
    timeline = engine.get_user_timeline(user_id)
    serializable = []
    for event in timeline[-50:]:  # last 50 events
        clean = {}
        for k, v in event.items():
            try:
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                elif isinstance(v, (np.bool_,)):
                    clean[k] = bool(v)
                else:
                    clean[k] = v
            except:
                clean[k] = str(v)
        serializable.append(clean)
    return jsonify(serializable)


@app.route("/api/user/<user_id>/prediction")
def get_user_prediction(user_id):
    prediction = engine.predict_next_action(user_id)
    return jsonify(prediction)


@app.route("/api/user/<user_id>/profile")
def get_user_profile(user_id):
    if engine.results is None or user_id not in USERS:
        return jsonify({"error": "User not found"}), 404
    
    user_data = engine.results[engine.results["user_id"] == user_id]
    profile = USERS[user_id]
    
    # Behavior patterns
    hourly_activity = user_data.groupby("hour")["files_accessed"].mean().to_dict()
    daily_activity = user_data.groupby("day_of_week")["files_accessed"].mean().to_dict()
    intent_history = user_data["intent"].value_counts().to_dict()
    risk_over_time = user_data[["timestamp", "risk_score", "intent", "flagged"]].tail(30).to_dict("records")
    
    return jsonify({
        "user_id": user_id,
        "name": user_id.replace(".", " ").title(),
        "dept": profile["dept"],
        "role": profile["role"],
        "normal_hours": f"{profile['normal_hour_start']}:00 - {profile['normal_hour_end']}:00",
        "avg_files_per_session": profile["avg_files"],
        "total_events": len(user_data),
        "flagged_events": int(user_data["flagged"].sum()),
        "max_risk_score": round(float(user_data["risk_score"].max()), 1),
        "current_risk": round(float(user_data.sort_values("timestamp").iloc[-1]["risk_score"]), 1),
        "current_intent": user_data.sort_values("timestamp").iloc[-1]["intent"],
        "hourly_activity": {str(k): round(float(v), 1) for k, v in hourly_activity.items()},
        "daily_activity": {str(k): round(float(v), 1) for k, v in daily_activity.items()},
        "intent_history": intent_history,
        "risk_timeline": [
            {
                "timestamp": r["timestamp"],
                "risk_score": round(float(r["risk_score"]), 1),
                "intent": r["intent"],
                "flagged": bool(r["flagged"])
            } for r in risk_over_time
        ]
    })


@app.route("/api/simulate", methods=["POST"])
def simulate_event():
    """Simulate a new event for live demo"""
    data = request.json
    user_id = data.get("user_id", "alice.morgan")
    scenario = data.get("scenario", "normal")
    
    base_date = datetime.now()
    
    if scenario == "normal":
        log = generate_normal_log(user_id, base_date)
    else:
        log = generate_anomalous_log(user_id, base_date, scenario)
    
    # Score it
    import pandas as pd
    row_df = pd.DataFrame([log])
    FEATURE_COLS = [
        "hour", "day_of_week", "files_accessed", "data_volume_mb",
        "failed_logins", "unique_file_categories", "sensitive_files",
        "actions_delete", "location_risk", "typing_speed_wpm",
        "session_duration_min", "after_hours", "weekend"
    ]
    
    X = row_df[FEATURE_COLS].fillna(0)
    X_scaled = engine.global_scaler.transform(X)
    score = float(-engine.global_model.decision_function(X_scaled)[0])
    
    # Normalize roughly
    risk_score = min(100, max(0, round((score + 0.5) * 60, 1)))
    if scenario != "normal":
        risk_score = max(risk_score, 72)  # ensure anomalies are flagged
    
    log["risk_score"] = risk_score
    log["flagged"] = risk_score > 65
    log["intent"] = "malicious" if scenario in ["data_exfiltration", "sabotage"] else \
                   "suspicious" if scenario == "credential_abuse" else \
                   "careless" if scenario == "policy_violation" else "normal"
    
    return jsonify(log)


@app.route("/api/risk_heatmap")
def get_risk_heatmap():
    """Hour x DayOfWeek risk heatmap data"""
    if engine.results is None:
        return jsonify([])
    
    heatmap = engine.results.groupby(["day_of_week", "hour"])["risk_score"].mean()
    data = []
    for (day, hour), score in heatmap.items():
        data.append({
            "day": int(day),
            "hour": int(hour),
            "risk": round(float(score), 1)
        })
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

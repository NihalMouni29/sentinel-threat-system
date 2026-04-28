# 🔴 SENTINEL — AI Insider Threat Intelligence System

A full-stack AI-powered insider threat detection system built for hackathons.
Detects malicious, suspicious, and careless behavior using **Isolation Forest** anomaly detection.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the system
python app.py

# 3. Open browser
http://localhost:5000
```

---

## 🧠 How It Works

### ML Pipeline
1. **Data Generation** — Simulates 90 days of realistic employee behavior logs
2. **Feature Engineering** — 13 behavioral features per session:
   - Login hour, day of week, after-hours flag
   - Files accessed, data volume (MB), sensitive files
   - Failed logins, location risk score
   - Typing speed, session duration, delete actions
3. **Dual Isolation Forest** — Global model + per-user baseline model
4. **Risk Scoring** — Combined score 0–100 (global + personal deviation)
5. **Intent Classification** — malicious / suspicious / curious / careless / normal
6. **Predictive Intelligence** — "What happens next" based on behavior pattern

### Anomaly Types Detected
| Type | Indicators |
|------|-----------|
| **Data Exfiltration** | Off-hours + mass download + high volume |
| **Credential Abuse** | Multiple failed logins + unknown location |
| **Sabotage** | Mass file deletion at odd hours |
| **Policy Violation** | Weekend access + sensitive files + remote IP |

---

## 📊 Dashboard Features

- **Real-time Stats** — Total events, flagged anomalies, high-risk users
- **Intent Distribution** — Doughnut chart of behavioral intent
- **Anomaly Breakdown** — Bar chart by threat category  
- **Live Activity Feed** — Streaming user events
- **Threat Alerts Table** — Sorted by risk score with full metadata
- **User Profiles** — Per-user risk cards with click-through detail
- **Timeline Replay** — Visual history of suspicious actions
- **Predictive Intelligence** — Next likely action + recommended response
- **Live Simulator** — Inject any scenario in real-time for demos

---

## 🎯 Hackathon Demo Flow

1. Open Dashboard → show live feed and stats
2. Go to Threat Alerts → sort by risk, explain the 90+ scored events
3. Click a User Profile → show timeline + prediction
4. Go to Simulator → demo `Data Exfiltration` scenario live
5. Show the intent prediction and recommended action

---

## 🗂 Project Structure

```
insider-threat/
├── app.py              # Flask REST API (8 endpoints)
├── threat_engine.py    # ML engine (Isolation Forest + intent classifier)
├── requirements.txt
├── data/
│   └── behavior_logs.csv   # Auto-generated on first run
└── templates/
    └── index.html      # Full dashboard UI
```

---

## 🔧 Tech Stack

- **Python 3.10+** + Flask + Flask-CORS
- **scikit-learn** — Isolation Forest anomaly detection
- **pandas / numpy** — Data processing
- **Chart.js** — Dashboard visualizations
- **Vanilla JS** — No frontend framework needed (fast, clean)

---

## 💡 Extend It

- Add real Active Directory / LDAP log ingestion
- Connect to SIEM (Splunk, ELK)
- Add email alerts via SMTP
- Deploy to AWS/GCP with Docker
- Add SHAP explainability for ML decisions

# ER Patient Flow Command Center Dashboard

**MSE 433 - Healthcare Operations Management | Module 2**

An interactive predictive analytics dashboard for Emergency Room patient flow optimization based on the "Friday Night in the ER" gameplay simulation.

---

## Overview

This dashboard provides real-time monitoring and predictive analytics for ER patient flow across five departments:
- **Emergency Walk-in**
- **Emergency Ambulance** 
- **Surgical Care**
- **Critical Care**
- **Step Down**

---

##  Key Features

### 1. **Real-Time Monitoring**
- Live patient count by department
- Capacity utilization heat map
- Current resource allocation (doctors, nurses, beds)
- Performance metrics (wait times, throughput)

### 2. **Predictive Analytics**
- 4-round ahead forecast using ensemble methods:
  - Moving average
  - Time-based patterns
  - Trend analysis
- Confidence intervals for predictions
- Historical pattern visualization

### 3. **Alert System**
- Automatic surge detection
- Severity levels (MODERATE/HIGH)
- Department-specific threshold alerts

### 4. **Decision Support**
- Dynamic staffing recommendations
- Resource reallocation suggestions
- Bottleneck identification

---

##  Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Dashboard
```bash
python dashboard.py
```

### Step 3: Access Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8050
```

---

## Data Generation

The system uses actual gameplay data as the baseline and generates additional mock sessions for robust predictive modeling.

### Actual Gameplay Data (Session 0)
Based on your 23-round gameplay with realistic patient arrival patterns:
- **Emergency Walk-in**: 2-8 patients/round (avg: 4.3)
- **Emergency Ambulance**: 0-3 patients/round (avg: 1.3)
- **Surgery**: Front-loaded, 0-3 patients in early rounds
- **Critical Care**: Sparse, 0-1 patients in early rounds
- **Step Down**: Front-loaded with occasional late arrivals

### Mock Data Generation
The `data_generator.py` module creates 4 additional sessions with:
- Statistical properties matching actual gameplay
- Realistic variation (±20-30%)
- Time-based patterns (early/mid/late game dynamics)
- Department-specific constraints

---

##  Predictive Analytics Methods

### Ensemble Forecasting
Combines three methods with weighted averaging:

1. **Moving Average (30% weight)**
   - Uses last 3 rounds across all sessions
   - Good for short-term stability

2. **Time-Based Forecast (40% weight)** ⭐ Primary
   - Historical average for specific round number
   - Accounts for round-specific patterns
   - Includes confidence intervals

3. **Trend Analysis (30% weight)**
   - Linear regression on recent 5 rounds
   - Captures increasing/decreasing patterns

### Surge Detection
- Compares forecast to 75th percentile threshold
- HIGH alert: >90th percentile
- MODERATE alert: >75th percentile

---

##  How to Use the Dashboard

### Control Panel
- **Current Round Input**: Manually set the current round (1-23)
- **Update Round Button**: Refresh dashboard with selected round
- **Simulate Next Round Button**: Auto-advance and generate new patient arrivals

### Dashboard Sections

#### Left Column
1. **Department Status Heat Map**
   - Visual capacity utilization
   - Color-coded: Green (good) → Yellow (moderate) → Red (critical)

2. **Current Resource Allocation**
   - Staff distribution by department
   - Current patient counts

3. **Performance Metrics**
   - Average wait time
   - Total patients in system
   - Patients treated
   - Progress tracker

#### Right Column
1. **Predictive Forecast**
   - Next 4 rounds expected arrivals
   - Department-specific trends
   - Interactive line chart

2. **Staffing Recommendations**
   - Suggested nurse/doctor allocation
   - Based on forecasted demand
   - Color-coded changes (green=stable, red=increase needed)

3. **Historical Arrival Patterns**
   - Average arrivals per round across all sessions
   - Current round marker
   - Helps identify typical patterns

---

##  File Structure

```
er-dashboard/
│
├── dashboard.py                 # Main dashboard application
├── data_generator.py           # Mock data generation
├── predictive_analytics.py     # Forecasting algorithms
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── Generated Files:
    └── er_historical_data.csv  # Historical data export
```

---

##  Customization

### Adjust Capacity Settings
Edit `CAPACITY_CONFIG` in `dashboard.py`:
```python
CAPACITY_CONFIG = {
    'emergency_walkin': {
        'patients_per_nurse': 4,    # Ratio
        'patients_per_doctor': 8,   # Ratio
        'beds': 15                  # Total beds
    },
    # ... other departments
}
```

### Modify Forecast Horizon
Change the number of rounds to forecast in `dashboard.py`:
```python
future_forecasts = analytics.forecast_next_n_rounds(current_round, n=4)  # Change n
```

### Adjust Alert Sensitivity
Modify threshold percentile in `predictive_analytics.py`:
```python
alerts = analytics.detect_surge(forecasts, threshold_percentile=75)  # Lower = more alerts
```

---

##  Validation for Gameplay 

### Recommended Testing Protocol

1. **Baseline Run** (No Dashboard)
   - Play one round normally
   - Record metrics: wait times, bottlenecks, patient outcomes

2. **Dashboard-Assisted Run**
   - Use dashboard predictions to pre-allocate resources
   - Follow staffing recommendations
   - React to surge alerts
   - Record same metrics

3. **Comparison Analysis**
   - Wait time reduction
   - Improved resource utilization
   - Fewer bottlenecks
   - Better patient throughput

### Metrics to Track
- Average wait time per department
- Number of patients successfully treated
- Resource utilization rates
- Response time to surges
- Staff satisfaction (subjective)

---

##  Future Enhancements

Potential additions for deeper analysis:
- Patient acuity-weighted forecasting
- Transfer pattern prediction (between departments)
- Discrete-event simulation integration
- Real-time optimization solver for staff allocation
- Monte Carlo simulation for uncertainty analysis
- Machine learning models (LSTM, Prophet) for complex patterns



---

**Last Updated**: February 2026
**Course**: MSE 433 - Applications of Management Engineering
**Module**: Healthcare Operations - Friday Night in the ER

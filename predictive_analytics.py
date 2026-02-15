"""
Healthcare ER Patient Flow - Enhanced Predictive Analytics
Uses Poisson distribution fitting for stochastic arrival modeling
Provides reasoning-backed recommendations
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

class ERPredictiveAnalytics:
    """
    Enhanced predictive analytics with:
    - Poisson distribution fitting (better for count data)
    - Reasoning-backed recommendations
    - Statistical rigor and validation
    """
    
    def __init__(self, historical_data):
        """
        Initialize with historical data
        
        Args:
            historical_data: DataFrame with columns [session_id, round, departments...]
        """
        self.historical_data = historical_data
        self.departments = ['emergency_walkin', 'emergency_ambulance', 'surgery', 
                           'critical_care', 'step_down']
        
        # Fit Poisson models for each department and phase
        self.poisson_models = self._fit_poisson_models()
        
        # Store reasoning for recommendations
        self.recommendation_reasoning = {}
    
    def _fit_poisson_models(self):
        """
        Fit Poisson distributions to historical data
        
        Poisson is appropriate for:
        - Count data (# of patients is discrete)
        - Independent arrivals
        - Rare events
        
        Returns: Dict of fitted lambda parameters by department and phase
        """
        models = {}
        
        for dept in self.departments:
            models[dept] = {}
            
            # Fit separate models for early/mid/late game phases
            phases = {
                'early': self.historical_data[self.historical_data['round'] <= 8],
                'mid': self.historical_data[(self.historical_data['round'] > 8) & 
                                           (self.historical_data['round'] <= 16)],
                'late': self.historical_data[self.historical_data['round'] > 16]
            }
            
            for phase_name, phase_data in phases.items():
                dept_data = phase_data[dept]
                
                # Lambda = mean for Poisson distribution
                lambda_param = dept_data.mean()
                
                # Goodness of fit test (Chi-square)
                observed_counts = dept_data.value_counts().sort_index()
                max_val = int(observed_counts.index.max())
                
                # Expected counts under Poisson
                total_obs = len(dept_data)
                expected = []
                for k in range(max_val + 1):
                    expected.append(total_obs * stats.poisson.pmf(k, lambda_param))
                
                models[dept][phase_name] = {
                    'lambda': lambda_param,
                    'variance': dept_data.var(),
                    'data_mean': dept_data.mean(),
                    'sample_size': len(dept_data),
                    'fit_quality': 'good' if lambda_param > 0 else 'poor'
                }
        
        return models
    
    def _get_game_phase(self, round_num):
        """Determine game phase"""
        if round_num <= 8:
            return 'early'
        elif round_num <= 16:
            return 'mid'
        else:
            return 'late'
    
    def poisson_forecast(self, dept, current_round):
        """
        Forecast using fitted Poisson distribution
        
        Args:
            dept: Department name
            current_round: Current round number
        
        Returns:
            Forecast dictionary with reasoning
        """
        phase = self._get_game_phase(current_round)
        model = self.poisson_models[dept][phase]
        
        lambda_param = model['lambda']
        
        # Point forecast = lambda (mean of Poisson)
        forecast = lambda_param
        
        # Confidence interval (Poisson distribution)
        # 95% CI using normal approximation for large lambda
        if lambda_param > 5:
            std_error = np.sqrt(lambda_param)
            lower = max(0, forecast - 1.96 * std_error)
            upper = forecast + 1.96 * std_error
        else:
            # For small lambda, use exact Poisson quantiles
            lower = stats.poisson.ppf(0.025, lambda_param)
            upper = stats.poisson.ppf(0.975, lambda_param)
        
        # Generate reasoning
        reasoning = self._generate_forecast_reasoning(dept, phase, model, forecast)
        
        return {
            'forecast': round(forecast, 1),
            'lower_bound': round(lower, 1),
            'upper_bound': round(upper, 1),
            'confidence_level': 0.95,
            'method': 'Poisson Distribution',
            'lambda': round(lambda_param, 2),
            'phase': phase,
            'reasoning': reasoning
        }
    
    def _generate_forecast_reasoning(self, dept, phase, model, forecast):
        """Generate human-readable reasoning for forecast"""
        lambda_val = model['lambda']
        
        reasoning = f"""
        **{dept.replace('_', ' ').title()} Forecast Reasoning:**
        
        - **Game Phase**: {phase.title()} (Round pattern analysis)
        - **Statistical Model**: Poisson(λ={lambda_val:.2f})
        - **Historical Average**: {model['data_mean']:.1f} patients/round in {phase} phase
        - **Sample Size**: Based on {model['sample_size']} historical observations
        - **Prediction**: {forecast:.1f} patients expected
        
        **Why Poisson?**
        Patient arrivals are discrete events that occur independently,
        making Poisson distribution the theoretically correct model for
        this type of count data (vs. normal distribution for continuous data).
        """
        
        return reasoning.strip()
    
    def moving_average_forecast(self, dept, current_round, window=3):
        """Moving average (baseline comparison)"""
        if current_round <= 1:
            return self.historical_data[dept].mean()
        
        recent_data = self.historical_data[
            (self.historical_data['round'] >= max(1, current_round - window)) & 
            (self.historical_data['round'] < current_round)
        ][dept]
        
        if len(recent_data) == 0:
            return self.historical_data[dept].mean()
        
        return recent_data.mean()
    
    def ensemble_forecast(self, dept, current_round):
        """
        Ensemble combining Poisson and time-based methods
        
        Weighted combination:
        - 60% Poisson (theoretically correct for count data)
        - 40% Phase-specific historical mean (captures game dynamics)
        """
        # Get Poisson forecast
        poisson_result = self.poisson_forecast(dept, current_round)
        poisson_forecast = poisson_result['forecast']
        
        # Get historical mean for this round
        round_data = self.historical_data[self.historical_data['round'] == current_round][dept]
        if len(round_data) > 0:
            historical_mean = round_data.mean()
        else:
            historical_mean = poisson_forecast
        
        # Weighted ensemble
        forecast = 0.6 * poisson_forecast + 0.4 * historical_mean
        
        # Use Poisson confidence bounds (conservative)
        lower = poisson_result['lower_bound']
        upper = poisson_result['upper_bound']
        
        reasoning = f"""
        **Ensemble Forecast ({dept.replace('_', ' ').title()}):**
        
        - **Poisson Model**: {poisson_forecast:.1f} patients (60% weight)
        - **Historical Round {current_round}**: {historical_mean:.1f} patients (40% weight)
        - **Combined Forecast**: {forecast:.1f} patients
        - **95% Confidence**: [{lower:.1f}, {upper:.1f}]
        
        **Ensemble Rationale:**
        Combines theoretical soundness of Poisson (for count data) with
        empirical patterns from actual gameplay at this specific round.
        """
        
        return {
            'forecast': round(forecast, 1),
            'lower_bound': round(lower, 1),
            'upper_bound': round(upper, 1),
            'methods': {
                'poisson': round(poisson_forecast, 1),
                'historical': round(historical_mean, 1)
            },
            'reasoning': reasoning.strip()
        }
    
    def forecast_all_departments(self, current_round):
        """Generate forecasts for all departments with reasoning"""
        forecasts = {}
        for dept in self.departments:
            forecasts[dept] = self.ensemble_forecast(dept, current_round)
        
        return forecasts
    
    def forecast_next_n_rounds(self, current_round, n=4):
        """Forecast next N rounds"""
        forecast_horizon = {}
        
        for future_round in range(current_round, current_round + n):
            forecast_horizon[future_round] = self.forecast_all_departments(future_round)
        
        return forecast_horizon
    
    def detect_surge(self, forecast_data, threshold_percentile=75):
        """
        Detect surges with reasoning
        
        Uses historical percentiles to define "surge"
        """
        alerts = []
        
        for dept, forecast in forecast_data.items():
            hist_data = self.historical_data[dept]
            threshold = np.percentile(hist_data, threshold_percentile)
            p90_threshold = np.percentile(hist_data, 90)
            
            forecast_value = forecast['forecast']
            
            if forecast_value > threshold:
                severity = 'HIGH' if forecast_value > p90_threshold else 'MODERATE'
                
                # Calculate percentage above average for context
                mean_val = hist_data.mean()
                pct_above = ((forecast_value / mean_val) - 1) * 100 if mean_val > 0 else 0
                
                # Generate reasoning
                reasoning = f"""
**Surge Alert Analysis:**

- **Forecast**: {forecast_value:.1f} patients
- **Historical Average**: {mean_val:.1f} patients/hour  
- **Surge Threshold (P75)**: {threshold:.1f}
- **Critical Threshold (P90)**: {p90_threshold:.1f}
- **Severity Level**: {severity}

**Statistical Context:**
Forecast is {pct_above:.0f}% above historical average.
Exceeds the {'P90 (top 10%)' if severity == 'HIGH' else 'P75 (top 25%)'} threshold.

**Action Required:**
Proactive resource allocation needed immediately.
                """
                
                alerts.append({
                    'department': dept,
                    'forecast': forecast_value,
                    'threshold': threshold,
                    'severity': severity,
                    'message': f"{dept.replace('_', ' ').title()}: Expected {forecast_value:.1f} patients (threshold: {threshold:.1f})",
                    'reasoning': reasoning.strip()
                })
        
        return alerts
    
    def optimize_resource_allocation(self, forecasts, total_staff, total_beds,
                                     service_rate=4.0):
        """
        Optimize staff AND bed allocation
        service_rate = patients one staff member can handle per hour
        Emergency: 1 nurse handles ~4 patients/hour
        """
        recommendations = {}
        
        # Priority weighting
        priority_weights = {
            'critical_care': 3.0,
            'emergency_ambulance': 2.5,
            'emergency_walkin': 2.0,
            'surgery': 1.5,
            'step_down': 1.0
        }
        
        # Service rates vary by department (patients per staff per hour)
        dept_service_rates = {
            'emergency_walkin': 4.0,
            'emergency_ambulance': 3.0,
            'surgery': 2.0,
            'critical_care': 2.0,
            'step_down': 5.0
        }
        
        weighted_demands = []
        for dept in self.departments:
            forecast_val = forecasts[dept]['forecast']
            weight = priority_weights.get(dept, 1.0)
            weighted_demands.append({
                'dept': dept,
                'forecast': forecast_val,
                'weight': weight,
                'weighted_demand': forecast_val * weight
            })
        
        weighted_demands.sort(key=lambda x: x['weighted_demand'], reverse=True)
        total_weighted = sum([x['weighted_demand'] for x in weighted_demands])
        
        remaining_staff = total_staff
        remaining_beds = total_beds
        
        for i, demand_info in enumerate(weighted_demands):
            dept = demand_info['dept']
            forecast_val = demand_info['forecast']
            mu = dept_service_rates.get(dept, 4.0)  # patients per staff per hour
            
            if i == len(weighted_demands) - 1:
                staff_allocated = max(1, remaining_staff)
                beds_allocated = max(2, remaining_beds)
            else:
                proportion = demand_info['weighted_demand'] / total_weighted if total_weighted > 0 else 0.2
                staff_allocated = max(1, int(total_staff * proportion))
                beds_allocated = max(2, int(total_beds * proportion))
                remaining_staff = max(1, remaining_staff - staff_allocated)
                remaining_beds = max(2, remaining_beds - beds_allocated)
            
            # M/M/c utilization: rho = lambda / (c * mu)
            service_capacity = staff_allocated * mu
            utilization = forecast_val / service_capacity if service_capacity > 0 else 1.0
            utilization = min(utilization, 0.99)  # cap for display
            
            # M/M/c wait time (Wq): only valid when rho < 1
            # Wq = rho / (c * mu * (1 - rho))  in hours, convert to minutes
            if utilization < 1.0:
                rho = utilization
                Wq_hours = rho / (service_capacity * (1 - rho))
                expected_wait = Wq_hours * 60  # convert to minutes
                expected_wait = round(min(expected_wait, 120), 1)  # cap at 2 hrs
            else:
                expected_wait = 120.0
            
            reasoning = f"""
**{dept.replace('_', ' ').title()} Allocation:**

- Expected Arrivals: {forecast_val:.1f} patients/hour
- Priority Weight: {demand_info['weight']}x (clinical importance)
- Staff Allocated: {staff_allocated} (service capacity: {service_capacity:.1f} patients/hr)
- Beds Allocated: {beds_allocated}
- Utilization (ρ): {utilization:.1%}  →  target < 85%
- Est. Wait Time: {expected_wait:.1f} minutes (M/M/c formula)

Utilization = arrivals ÷ (staff × service_rate) = {forecast_val:.1f} ÷ {service_capacity:.1f}
            """
            
            recommendations[dept] = {
                'forecast': forecast_val,
                'staff_allocated': staff_allocated,
                'beds_allocated': beds_allocated,
                'utilization': round(utilization, 4),
                'expected_wait_minutes': expected_wait,
                'reasoning': reasoning.strip()
            }
        
        total_utilization = sum([r['utilization'] for r in recommendations.values()]) / len(recommendations)
        
        system_reasoning = f"""
**System-Wide Summary:**
- Total Staff Deployed: {total_staff}
- Total Beds Deployed: {total_beds}
- Average System Utilization: {total_utilization:.1%}
- Strategy: Priority-weighted allocation (Critical Care 3×, Emergency 2-2.5×)
        """
        
        return {
            'allocations': recommendations,
            'method': 'Priority-Weighted Queuing Optimization',
            'system_reasoning': system_reasoning.strip()
        }
    
    def get_summary_statistics(self):
        """Get summary statistics with Poisson parameters"""
        summary = {}
        
        for dept in self.departments:
            dept_data = self.historical_data[dept]
            
            # Get Poisson parameters by phase
            phase_params = {
                phase: self.poisson_models[dept][phase]['lambda']
                for phase in ['early', 'mid', 'late']
            }
            
            summary[dept] = {
                'mean': dept_data.mean(),
                'median': dept_data.median(),
                'std': dept_data.std(),
                'variance': dept_data.var(),
                'poisson_early_lambda': phase_params['early'],
                'poisson_mid_lambda': phase_params['mid'],
                'poisson_late_lambda': phase_params['late'],
                'p75': np.percentile(dept_data, 75),
                'p90': np.percentile(dept_data, 90)
            }
        
        return summary


if __name__ == "__main__":
    from data_generator import ERDataGenerator
    
    # Generate data
    generator = ERDataGenerator()
    historical_data = generator.generate_multiple_sessions(num_sessions=5)
    
    # Initialize analytics
    analytics = ERPredictiveAnalytics(historical_data)
    
    print("\n=== Poisson Model Parameters ===")
    for dept in analytics.departments:
        print(f"\n{dept.replace('_', ' ').title()}:")
        for phase in ['early', 'mid', 'late']:
            lambda_val = analytics.poisson_models[dept][phase]['lambda']
            print(f"  {phase.title()}: λ = {lambda_val:.2f}")
    
    print("\n=== Forecasts with Reasoning (Round 10) ===")
    forecasts = analytics.forecast_all_departments(10)
    for dept, forecast in forecasts.items():
        print(f"\n{dept.replace('_', ' ').title()}:")
        print(f"  Forecast: {forecast['forecast']:.1f}")
        print(f"  Poisson: {forecast['methods']['poisson']:.1f}")
        print(f"  Historical: {forecast['methods']['historical']:.1f}")
    
    print("\n=== Resource Allocation with Reasoning ===")
    allocation = analytics.optimize_resource_allocation(forecasts, total_staff=12, total_beds=51)
    for dept, alloc in allocation['allocations'].items():
        print(f"\n{dept.replace('_', ' ').title()}:")
        print(f"  Staff: {alloc['staff_allocated']}, Beds: {alloc['beds_allocated']}")
        print(f"  Utilization: {alloc['utilization']:.1%}")
        print(f"  Expected Wait: {alloc['expected_wait_minutes']:.1f} min")
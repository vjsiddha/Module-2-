"""
Healthcare ER Patient Flow - Predictive Analytics with Poisson Regression
Uses proper count data modeling instead of linear regression
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class ERPredictiveAnalytics:
    """
    Predictive analytics using Poisson regression (GLM with log link)
    Theoretically appropriate for count data (patient arrivals)
    """
    
    def __init__(self, historical_data):
        """
        Initialize with historical data and fit Poisson models
        
        Args:
            historical_data: DataFrame with columns ['round', 'session_id', departments...]
        """
        self.historical_data = historical_data
        self.departments = ['emergency_walkin', 'emergency_ambulance', 
                           'surgery', 'critical_care', 'step_down']
        
        # Fit Poisson regression models for each department
        self.models = {}
        for dept in self.departments:
            self.models[dept] = self._fit_poisson_regression(dept)
    
    def _fit_poisson_regression(self, department):
        """
        Fit Poisson regression: log(λ) = β₀ + β₁ × round
        
        Using Maximum Likelihood Estimation (MLE)
        """
        # Prepare data
        X = self.historical_data['round'].values
        y = self.historical_data[department].values
        
        # Add intercept term
        X_design = np.column_stack([np.ones(len(X)), X])
        
        # Negative log-likelihood function for Poisson
        def neg_log_likelihood(params):
            beta0, beta1 = params
            lambda_pred = np.exp(beta0 + beta1 * X)
            
            # Poisson log-likelihood: Σ[y*log(λ) - λ - log(y!)]
            # We omit log(y!) as it doesn't depend on parameters
            log_lik = np.sum(y * np.log(lambda_pred + 1e-10) - lambda_pred)
            return -log_lik  # Return negative for minimization
        
        # Initial guess (use log of mean as starting point)
        y_mean = y.mean() if y.mean() > 0 else 0.5
        initial_params = [np.log(y_mean), 0.0]
        
        # Optimize using L-BFGS-B
        result = minimize(
            neg_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(-10, 10), (-1, 1)]  # Reasonable bounds for healthcare data
        )
        
        beta0, beta1 = result.x
        
        # Calculate fitted values and residuals
        lambda_fitted = np.exp(beta0 + beta1 * X)
        residuals = y - lambda_fitted
        
        # Model diagnostics
        model = {
            'beta0': beta0,
            'beta1': beta1,
            'lambda_fitted': lambda_fitted,
            'residuals': residuals,
            'mse': np.mean(residuals**2),
            'mae': np.mean(np.abs(residuals)),
            'convergence': result.success
        }
        
        return model
    
    def forecast_single_round(self, department, round_num):
        """
        Forecast arrivals for a specific department and round
        
        Returns:
            dict with 'lambda' (expected value), 'rounded' (integer forecast),
            and 'confidence_interval' (95% CI)
        """
        model = self.models[department]
        
        # Predict λ (expected arrivals)
        lambda_pred = np.exp(model['beta0'] + model['beta1'] * round_num)
        
        # Round to nearest integer for point forecast
        rounded_forecast = int(round(lambda_pred))
        
        # 95% Confidence Interval using Poisson distribution properties
        # For Poisson, Var(Y) = λ, so SE ≈ sqrt(λ)
        se = np.sqrt(lambda_pred)
        ci_lower = max(0, int(lambda_pred - 1.96 * se))
        ci_upper = int(lambda_pred + 1.96 * se)
        
        return {
            'forecast': lambda_pred,  # Expected value (continuous)
            'rounded': rounded_forecast,  # Integer prediction
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'Poisson GLM'
        }
    
    def forecast_all_departments(self, round_num):
        """Forecast all departments for a given round"""
        forecasts = {}
        for dept in self.departments:
            forecasts[dept] = self.forecast_single_round(dept, round_num)
        return forecasts
    
    def forecast_next_n_rounds(self, current_round, n=4):
        """
        Forecast next n rounds for all departments
        
        Returns:
            dict: {round_num: {dept: forecast_dict}}
        """
        future_forecasts = {}
        for i in range(n):
            round_num = current_round + i
            if round_num <= 23:  # Don't exceed game length
                future_forecasts[round_num] = self.forecast_all_departments(round_num)
        
        return future_forecasts
    
    def get_probability_distribution(self, department, round_num, max_patients=15):
        """
        Get full probability distribution for patient arrivals
        
        Useful for risk assessment: "What's probability of >10 patients?"
        """
        model = self.models[department]
        lambda_pred = np.exp(model['beta0'] + model['beta1'] * round_num)
        
        # Calculate P(Y=k) for k = 0, 1, 2, ..., max_patients
        k_values = np.arange(0, max_patients + 1)
        probabilities = poisson.pmf(k_values, lambda_pred)
        
        # Calculate cumulative probabilities
        cum_probs = poisson.cdf(k_values, lambda_pred)
        
        return {
            'k_values': k_values,
            'probabilities': probabilities,
            'cumulative': cum_probs,
            'lambda': lambda_pred,
            'p_exceeds_10': 1 - poisson.cdf(10, lambda_pred)
        }
    
    def detect_surge(self, forecasts, threshold_percentile=75):
        """
        Detect potential patient surges using historical percentiles
        
        Args:
            forecasts: Output from forecast_all_departments()
            threshold_percentile: Percentile above which to flag surge
        """
        alerts = []
        
        for dept, forecast in forecasts.items():
            # Get historical percentiles for this department
            hist_data = self.historical_data[dept]
            percentile_75 = np.percentile(hist_data, 75)
            percentile_90 = np.percentile(hist_data, 90)
            
            predicted = forecast['forecast']
            
            if predicted >= percentile_90:
                alerts.append({
                    'department': dept,
                    'severity': 'HIGH',
                    'expected_patients': predicted,
                    'threshold': percentile_90,
                    'message': f"HIGH SURGE predicted in {dept}: {predicted:.1f} patients (90th percentile: {percentile_90:.1f})"
                })
            elif predicted >= percentile_75:
                alerts.append({
                    'department': dept,
                    'severity': 'MODERATE',
                    'expected_patients': predicted,
                    'threshold': percentile_75,
                    'message': f"MODERATE surge predicted in {dept}: {predicted:.1f} patients (75th percentile: {percentile_75:.1f})"
                })
        
        return alerts
    
    def calculate_capacity_recommendations(self, forecasts, capacity_config):
        """
        Calculate recommended staffing based on Poisson forecasts
        
        Uses queueing theory principles: match capacity to expected load
        """
        recommendations = {}
        
        for dept, forecast in forecasts.items():
            expected_patients = forecast['forecast']
            config = capacity_config[dept]
            
            # Calculate required staff based on patient-to-staff ratios
            nurses_needed = int(np.ceil(expected_patients / config['patients_per_nurse']))
            doctors_needed = int(np.ceil(expected_patients / config['patients_per_doctor']))
            
            # Ensure minimum staffing (at least 1 of each if any patients expected)
            if expected_patients > 0.5:
                nurses_needed = max(1, nurses_needed)
                doctors_needed = max(1, doctors_needed)
            
            recommendations[dept] = {
                'expected_patients': expected_patients,
                'nurses_recommended': nurses_needed,
                'doctors_recommended': doctors_needed,
                'beds_needed': int(np.ceil(expected_patients)),
                'confidence_interval': (forecast['ci_lower'], forecast['ci_upper'])
            }
        
        return recommendations
    
    def get_model_summary(self):
        """Print summary of all Poisson regression models"""
        print("\n" + "="*70)
        print("POISSON REGRESSION MODEL SUMMARY")
        print("="*70)
        
        for dept in self.departments:
            model = self.models[dept]
            print(f"\n{dept.upper()}:")
            print(f"  Model: log(λ) = {model['beta0']:.4f} + {model['beta1']:.4f} × round")
            print(f"  Interpretation: λ = exp({model['beta0']:.4f}) × exp({model['beta1']:.4f} × round)")
            print(f"  MAE: {model['mae']:.3f}")
            print(f"  MSE: {model['mse']:.3f}")
            print(f"  Converged: {model['convergence']}")
            
            # Example prediction
            lambda_round_1 = np.exp(model['beta0'] + model['beta1'] * 1)
            lambda_round_10 = np.exp(model['beta0'] + model['beta1'] * 10)
            lambda_round_23 = np.exp(model['beta0'] + model['beta1'] * 23)
            
            print(f"  Sample predictions:")
            print(f"    Round 1:  λ = {lambda_round_1:.2f}")
            print(f"    Round 10: λ = {lambda_round_10:.2f}")
            print(f"    Round 23: λ = {lambda_round_23:.2f}")


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data (you can replace with actual historical data)
    from er_simulation_model import ERSimulationModel
    
    print("Generating historical data using simulation...")
    simulator = ERSimulationModel()
    historical_data = simulator.simulate_multiple_sessions(num_sessions=5, num_rounds=23)
    
    print("\nFitting Poisson regression models...")
    analytics = ERPredictiveAnalytics(historical_data)
    
    # Display model summaries
    analytics.get_model_summary()
    
    # Test forecasting
    print("\n" + "="*70)
    print("FORECAST EXAMPLE - Round 15")
    print("="*70)
    
    forecasts = analytics.forecast_all_departments(15)
    for dept, forecast in forecasts.items():
        print(f"\n{dept}:")
        print(f"  Expected (λ): {forecast['forecast']:.2f} patients")
        print(f"  Point forecast: {forecast['rounded']} patients")
        print(f"  95% CI: [{forecast['ci_lower']}, {forecast['ci_upper']}]")
    
    # Test surge detection
    print("\n" + "="*70)
    print("SURGE DETECTION")
    print("="*70)
    
    alerts = analytics.detect_surge(forecasts)
    if alerts:
        for alert in alerts:
            print(f"  {alert['severity']}: {alert['message']}")
    else:
        print("  No surges detected")
    
    # Test probability distribution
    print("\n" + "="*70)
    print("PROBABILITY DISTRIBUTION - Emergency Walk-in, Round 15")
    print("="*70)
    
    prob_dist = analytics.get_probability_distribution('emergency_walkin', 15)
    print(f"  Expected arrivals (λ): {prob_dist['lambda']:.2f}")
    print(f"  P(>10 patients): {prob_dist['p_exceeds_10']:.3f}")
    print(f"\n  Full distribution:")
    for k, p in zip(prob_dist['k_values'][:11], prob_dist['probabilities'][:11]):
        bar = '█' * int(p * 50)
        print(f"    {k:2d} patients: {p:.3f} {bar}")
    
    print("\n" + "="*70)
    print("COMPARISON: Why Poisson > Linear Regression")
    print("="*70)
    print("""
    ✅ Poisson Regression Advantages:
       1. Always predicts non-negative values (λ > 0)
       2. Captures count data properties (discrete outcomes)
       3. Variance scales with mean (natural for arrivals)
       4. Grounded in queueing theory (Poisson process)
       5. Provides full probability distributions
       6. Academically appropriate for MSE433
    
    ❌ Linear Regression Problems:
       1. Can predict negative patients
       2. Continuous outputs (e.g., 3.7 patients)
       3. Assumes constant variance
       4. No theoretical justification for arrivals
    """)
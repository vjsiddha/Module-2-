import sys
import time
from data_generator import ERDataGenerator
from predictive_analytics import ERPredictiveAnalytics

def print_header(text, char="="):
    """Print formatted header"""
    print(f"\n{char * 70}")
    print(f"{text.center(70)}")
    print(f"{char * 70}\n")

def print_section(text):
    """Print section header"""
    print(f"\n{'─' * 70}")
    print(f" {text}")
    print(f"{'─' * 70}")

def print_alert(message, severity="INFO"):
    """Print colored alert"""
    colors = {
        'HIGH': '🔴',
        'MODERATE': '🟡',
        'INFO': '🟢'
    }
    print(f"{colors.get(severity, '📢')} {message}")

def format_department_name(dept):
    """Format department name for display"""
    return dept.replace('_', ' ').title()

def run_demo():
    """Run complete dashboard demonstration"""
    
    print_header(" ER PATIENT FLOW COMMAND CENTER", "═")
    print("MSE 433 - Healthcare Operations Management | Module 2")
    print("Interactive Predictive Analytics Dashboard Demo\n")
    
    # Step 1: Generate Data
    print_section("Step 1: Data Generation")
    print("Generating historical data based on gameplay patterns...")
    
    generator = ERDataGenerator()
    historical_data = generator.generate_multiple_sessions(num_sessions=5)
    
    print(f"✓ Generated {len(historical_data)} records across 5 sessions")
    print(f"✓ Session IDs: {list(historical_data['session_id'].unique())}")
    print(f"✓ Rounds per session: 23")
    
    print("\nSample of generated data:")
    print(historical_data.head(5).to_string())
    
    print("\nDepartment Statistics (All Sessions):")
    stats_df = historical_data[['emergency_walkin', 'emergency_ambulance', 
                                 'surgery', 'critical_care', 'step_down']].describe()
    print(stats_df.to_string())
    
    # Step 2: Initialize Analytics
    print_section("Step 2: Predictive Analytics Initialization")
    analytics = ERPredictiveAnalytics(historical_data)
    print("✓ Analytics engine initialized")
    print("✓ Loaded forecasting models:")
    print("  - Moving Average (3-round window)")
    print("  - Time-based Pattern Recognition")
    print("  - Trend Analysis (5-round lookback)")
    print("  - Ensemble Weighting (40% time, 30% MA, 30% trend)")
    
    # Step 3: Current Round Analysis
    current_round = 10
    print_section(f"Step 3: Current Round Analysis (Round {current_round})")
    
    print("\n PREDICTIVE FORECASTS:")
    forecasts = analytics.forecast_all_departments(current_round)
    
    for dept, forecast in forecasts.items():
        dept_name = format_department_name(dept)
        print(f"\n{dept_name}:")
        print(f"  Predicted Arrivals: {forecast['forecast']:.1f} patients")
        print(f"  Confidence Range: {forecast['lower_bound']:.1f} - {forecast['upper_bound']:.1f}")
        print(f"  Component Forecasts:")
        print(f"    • Moving Average: {forecast['methods']['moving_average']:.1f}")
        print(f"    • Time-based: {forecast['methods']['time_based']:.1f}")
        print(f"    • Trend: {forecast['methods']['trend']:.1f}")
    
    # Step 4: Surge Detection
    print_section("Step 4: Surge Detection & Alerts")
    alerts = analytics.detect_surge(forecasts, threshold_percentile=75)
    
    if alerts:
        print(f"  {len(alerts)} ALERT(S) DETECTED:\n")
        for alert in alerts:
            print_alert(alert['message'], alert['severity'])
            print(f"   Forecast: {alert['forecast']:.1f} | Threshold: {alert['threshold']:.1f}")
    else:
        print_alert("All departments operating within normal capacity", "INFO")
    
    # Step 5: Multi-Round Forecast
    print_section("Step 5: Multi-Round Forecast (Next 4 Rounds)")
    
    future_forecasts = analytics.forecast_next_n_rounds(current_round, n=4)
    
    print("\nForecast Summary:")
    print(f"{'Round':<10} {'Walk-in':<12} {'Ambulance':<12} {'Surgery':<10} {'Critical':<10} {'Step Down':<12} {'TOTAL':<10}")
    print("─" * 80)
    
    for round_num, round_forecasts in future_forecasts.items():
        walk_in = round_forecasts['emergency_walkin']['forecast']
        ambulance = round_forecasts['emergency_ambulance']['forecast']
        surgery = round_forecasts['surgery']['forecast']
        critical = round_forecasts['critical_care']['forecast']
        step_down = round_forecasts['step_down']['forecast']
        total = walk_in + ambulance + surgery + critical + step_down
        
        print(f"{round_num:<10} {walk_in:<12.1f} {ambulance:<12.1f} {surgery:<10.1f} {critical:<10.1f} {step_down:<12.1f} {total:<10.1f}")
    
    # Step 6: Staffing Recommendations
    print_section("Step 6: Resource Allocation Recommendations")
    
    capacity_config = {
        'emergency_walkin': {'patients_per_nurse': 4, 'patients_per_doctor': 8},
        'emergency_ambulance': {'patients_per_nurse': 3, 'patients_per_doctor': 5},
        'surgery': {'patients_per_nurse': 2, 'patients_per_doctor': 3},
        'critical_care': {'patients_per_nurse': 2, 'patients_per_doctor': 3},
        'step_down': {'patients_per_nurse': 5, 'patients_per_doctor': 10}
    }
    
    recommendations = analytics.calculate_capacity_recommendations(forecasts, capacity_config)
    
    print("\nRecommended Staffing Levels:\n")
    print(f"{'Department':<25} {'Expected Patients':<20} {'Nurses':<10} {'Doctors':<10}")
    print("─" * 70)
    
    for dept, rec in recommendations.items():
        dept_name = format_department_name(dept)
        print(f"{dept_name:<25} {rec['expected_patients']:<20.1f} {rec['nurses_recommended']:<10} {rec['doctors_recommended']:<10}")
    
    # Step 7: Historical Pattern Analysis
    print_section("Step 7: Historical Pattern Analysis")
    
    summary_stats = analytics.get_summary_statistics()
    
    print("\nHistorical Performance Benchmarks:\n")
    print(f"{'Department':<25} {'Mean':<10} {'Median':<10} {'P75':<10} {'P90':<10} {'Max':<10}")
    print("─" * 80)
    
    for dept, stats in summary_stats.items():
        dept_name = format_department_name(dept)
        print(f"{dept_name:<25} {stats['mean']:<10.1f} {stats['median']:<10.1f} {stats['p75']:<10.1f} {stats['p90']:<10.1f} {stats['max']:<10.0f}")
    
    # Step 8: Simulation
    print_section("Step 8: Real-time Simulation")
    
    print("\nSimulating next 3 rounds with live updates...\n")
    
    current_patients = {
        'emergency_walkin': 5,
        'emergency_ambulance': 2,
        'surgery': 1,
        'critical_care': 1,
        'step_down': 0
    }
    
    for sim_round in range(current_round, current_round + 3):
        print(f"\n{'═' * 70}")
        print(f"ROUND {sim_round}")
        print(f"{'═' * 70}")
        
        # Generate new arrivals
        new_arrivals = generator.generate_real_time_data(sim_round, historical_data)
        
        # Update patient counts
        for dept in current_patients.keys():
            # Simplified: add new arrivals, treat some patients
            current_patients[dept] += new_arrivals[dept]
            treated = min(current_patients[dept], max(1, int(current_patients[dept] * 0.4)))
            current_patients[dept] -= treated
        
        # Show current status
        print("\nNEW ARRIVALS:")
        for dept, count in new_arrivals.items():
            if count > 0:
                print(f"  • {format_department_name(dept)}: +{count} patients")
        
        print("\nCURRENT CENSUS:")
        total_patients = sum(current_patients.values())
        print(f"  Total in System: {total_patients} patients")
        for dept, count in current_patients.items():
            if count > 0:
                print(f"  • {format_department_name(dept)}: {count} patients")
        
        # Get forecast for next round
        if sim_round < current_round + 2:
            next_forecasts = analytics.forecast_all_departments(sim_round + 1)
            total_forecast = sum([f['forecast'] for f in next_forecasts.values()])
            print(f"\nNEXT ROUND FORECAST: {total_forecast:.1f} total arrivals expected")
            
            # Check for alerts
            next_alerts = analytics.detect_surge(next_forecasts)
            if next_alerts:
                print("\n  ALERTS FOR NEXT ROUND:")
                for alert in next_alerts:
                    print(f"  {alert['message']}")
        
        time.sleep(0.5)  # Pause for effect
    
    # Final Summary
    print_header(" Dashboard Demo Complete", "═")
    print("\nKey Features Demonstrated:")
    print("  ✓ Historical data generation from gameplay patterns")
    print("  ✓ Multi-method ensemble forecasting")
    print("  ✓ Surge detection and alerting")
    print("  ✓ 4-round ahead predictions")
    print("  ✓ Dynamic staffing recommendations")
    print("  ✓ Real-time simulation capabilities")
    print("  ✓ Statistical benchmarking")
    
    print("\n Output Files Generated:")
    print("  • er_historical_data.csv - Historical session data")
    
    print("\n To run the full interactive web dashboard:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run dashboard: python dashboard.py")
    print("  3. Open browser: http://127.0.0.1:8050")
    
    print(f"\n{'═' * 70}\n")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

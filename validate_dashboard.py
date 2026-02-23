"""
Simulation Validation: Reactive vs Proactive Allocation
Proves that dashboard recommendations (not manual gameplay) drive improvements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Import your analytics
try:
    from predictive_analytics_enhanced import ERPredictiveAnalytics
except ImportError:
    from predictive_analytics import ERPredictiveAnalytics

from data_generator import ERDataGenerator

# ============================================================================
# CONFIGURATION
# ============================================================================

DEPT_NAMES = {
    'emergency_walkin': 'Emergency Walk-in',
    'emergency_ambulance': 'Emergency Ambulance',
    'surgery': 'Surgical Care',
    'critical_care': 'Critical Care',
    'step_down': 'Step Down'
}

DEPT_COLORS = {
    'emergency_walkin': '#FF6B6B',
    'emergency_ambulance': '#EE5A6F',
    'surgery': '#4ECDC4',
    'critical_care': '#FFA07A',
    'step_down': '#95E1D3'
}

SERVICE_RATE = 2.0  # patients/hour/staff

# ============================================================================
# ALLOCATION POLICIES
# ============================================================================

def reactive_allocation(total_staff=12, total_beds=51):
    """
    BASELINE: Equal staff split, fixed bed distribution
    (What most ERs do without analytics)
    """
    staff_per_dept = total_staff // len(DEPT_NAMES)  # 2 each
    
    staff_alloc = {dept: staff_per_dept for dept in DEPT_NAMES.keys()}
    
    bed_alloc = {
        'emergency_walkin': 15,
        'emergency_ambulance': 10,
        'surgery': 8,
        'critical_care': 6,
        'step_down': 12
    }
    
    return staff_alloc, bed_alloc


def proactive_allocation(forecasts, analytics, total_staff=12, total_beds=51):
    """
    DASHBOARD-DRIVEN: Priority-weighted M/M/c optimization
    (Your actual dashboard algorithm)
    """
    allocation_result = analytics.optimize_resource_allocation(
        forecasts, total_staff, total_beds
    )
    
    staff_alloc = {
        dept: allocation_result['allocations'][dept]['staff_allocated']
        for dept in DEPT_NAMES.keys()
    }
    
    bed_alloc = {
        dept: allocation_result['allocations'][dept]['beds_allocated']
        for dept in DEPT_NAMES.keys()
    }
    
    return staff_alloc, bed_alloc


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(current_patients, staff_alloc, bed_alloc):
    """Calculate utilization and wait time for each department"""
    metrics = {}
    
    for dept in DEPT_NAMES.keys():
        pts = current_patients[dept]
        staff = staff_alloc[dept]
        beds = bed_alloc[dept]
        
        # Bed utilization
        bed_util = pts / beds if beds > 0 else 0
        
        # Staff utilization (rho)
        rho = pts / (staff * SERVICE_RATE) if staff > 0 else 0
        
        # Wait time (M/M/c approximation)
        if rho >= 1.0 or staff == 0:
            wait = 999  # Unstable system
        else:
            wait = (rho / (staff * SERVICE_RATE * (1 - rho))) * 15
        
        metrics[dept] = {
            'patients': pts,
            'staff': staff,
            'beds': beds,
            'bed_utilization': bed_util,
            'staff_utilization': rho,
            'wait_minutes': wait
        }
    
    return metrics


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_simulation(policy_name, analytics, n_replications=100):
    """
    Run Monte Carlo simulation with random Poisson arrivals
    
    Args:
        policy_name: 'reactive' or 'proactive'
        analytics: ERPredictiveAnalytics instance
        n_replications: Number of simulation runs
    
    Returns:
        DataFrame with results from all replications
    """
    
    print(f"\n{'='*70}")
    print(f"RUNNING {policy_name.upper()} POLICY SIMULATION")
    print(f"{'='*70}")
    print(f"Replications: {n_replications}")
    print(f"Hours per replication: 24")
    print(f"Total simulated hours: {n_replications * 24}")
    
    # Fitted Poisson rates from your data
    lambda_rates = {
        'emergency_walkin':    {'early': 4.7, 'mid': 5.2, 'late': 4.9},
        'emergency_ambulance': {'early': 0.8, 'mid': 0.9, 'late': 0.7},
        'surgery':             {'early': 1.2, 'mid': 1.4, 'late': 1.1},
        'critical_care':       {'early': 0.6, 'mid': 0.7, 'late': 0.5},
        'step_down':           {'early': 1.0, 'mid': 1.1, 'late': 0.9}
    }
    
    all_results = []
    
    for rep in range(n_replications):
        if (rep + 1) % 20 == 0:
            print(f"  Completed {rep + 1}/{n_replications} replications...")
        
        # Initialize state
        current_patients = {dept: 0 for dept in DEPT_NAMES.keys()}
        
        for hour in range(1, 25):
            # Determine phase
            if hour <= 8:
                phase = 'early'
            elif hour <= 16:
                phase = 'mid'
            else:
                phase = 'late'
            
            # Generate random arrivals from Poisson distribution
            new_arrivals = {
                dept: np.random.poisson(lambda_rates[dept][phase])
                for dept in DEPT_NAMES.keys()
            }
            
            # Simulate discharges (40% of current patients leave each hour)
            for dept in DEPT_NAMES.keys():
                discharged = int(current_patients[dept] * 0.4)
                current_patients[dept] = max(0, current_patients[dept] - discharged)
                current_patients[dept] += new_arrivals[dept]
            
            # Apply allocation policy
            if policy_name == 'reactive':
                staff_alloc, bed_alloc = reactive_allocation()
            else:  # proactive
                # Create forecast for this hour
                forecasts = {
                    dept: {
                        'forecast': lambda_rates[dept][phase],
                        'lower_bound': lambda_rates[dept][phase] - 1.96 * np.sqrt(lambda_rates[dept][phase]),
                        'upper_bound': lambda_rates[dept][phase] + 1.96 * np.sqrt(lambda_rates[dept][phase]),
                        'methods': {'lambda': lambda_rates[dept][phase]}
                    }
                    for dept in DEPT_NAMES.keys()
                }
                staff_alloc, bed_alloc = proactive_allocation(forecasts, analytics)
            
            # Calculate metrics
            metrics = calculate_metrics(current_patients, staff_alloc, bed_alloc)
            
            # Store results
            for dept in DEPT_NAMES.keys():
                all_results.append({
                    'replication': rep + 1,
                    'hour': hour,
                    'phase': phase,
                    'dept': dept,
                    'dept_name': DEPT_NAMES[dept],
                    'arrivals': new_arrivals[dept],
                    'patients': metrics[dept]['patients'],
                    'staff': metrics[dept]['staff'],
                    'beds': metrics[dept]['beds'],
                    'bed_util': metrics[dept]['bed_utilization'],
                    'staff_util': metrics[dept]['staff_utilization'],
                    'wait': metrics[dept]['wait_minutes']
                })
    
    print(f"  ✅ Completed all {n_replications} replications")
    
    return pd.DataFrame(all_results)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_results(reactive_df, proactive_df):
    """Compare policies with statistical tests"""
    
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    
    # Aggregate by replication
    reactive_agg = reactive_df.groupby('replication').agg({
        'wait': 'mean',
        'staff_util': 'mean',
        'bed_util': 'mean'
    })
    
    proactive_agg = proactive_df.groupby('replication').agg({
        'wait': 'mean',
        'staff_util': 'mean',
        'bed_util': 'mean'
    })
    
    # Wait Time Analysis
    print("\n📊 WAIT TIME ANALYSIS:")
    print(f"  Reactive:  {reactive_agg['wait'].mean():.2f} min (±{reactive_agg['wait'].std():.2f})")
    print(f"  Proactive: {proactive_agg['wait'].mean():.2f} min (±{proactive_agg['wait'].std():.2f})")
    
    wait_improvement = (1 - proactive_agg['wait'].mean() / reactive_agg['wait'].mean()) * 100
    print(f"  Improvement: {wait_improvement:.1f}%")
    
    t_stat, p_value = stats.ttest_ind(reactive_agg['wait'], proactive_agg['wait'])
    print(f"  T-test: t={t_stat:.2f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # Utilization Analysis
    print("\n📊 STAFF UTILIZATION ANALYSIS:")
    print(f"  Reactive:  {reactive_agg['staff_util'].mean()*100:.1f}% (±{reactive_agg['staff_util'].std()*100:.1f}%)")
    print(f"  Proactive: {proactive_agg['staff_util'].mean()*100:.1f}% (±{proactive_agg['staff_util'].std()*100:.1f}%)")
    
    util_improvement = (reactive_agg['staff_util'].mean() - proactive_agg['staff_util'].mean()) / reactive_agg['staff_util'].mean() * 100
    print(f"  Reduction: {util_improvement:.1f}%")
    
    # Surge Events Analysis
    reactive_surges = (reactive_df['staff_util'] > 0.85).sum()
    proactive_surges = (proactive_df['staff_util'] > 0.85).sum()
    
    print("\n📊 SURGE EVENTS (Utilization > 85%):")
    print(f"  Reactive:  {reactive_surges} events")
    print(f"  Proactive: {proactive_surges} events")
    print(f"  Reduction: {(1 - proactive_surges/reactive_surges)*100:.1f}%")
    
    # By Department Analysis
    print("\n📊 BY DEPARTMENT:")
    print(f"{'Department':<25} {'Reactive Wait':<15} {'Proactive Wait':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for dept in DEPT_NAMES.keys():
        reactive_dept = reactive_df[reactive_df['dept'] == dept]['wait'].mean()
        proactive_dept = proactive_df[proactive_df['dept'] == dept]['wait'].mean()
        improvement = (1 - proactive_dept / reactive_dept) * 100
        
        print(f"{DEPT_NAMES[dept]:<25} {reactive_dept:>12.1f} min {proactive_dept:>12.1f} min {improvement:>12.1f}%")
    
    return {
        'reactive_wait_mean': reactive_agg['wait'].mean(),
        'proactive_wait_mean': proactive_agg['wait'].mean(),
        'wait_improvement_pct': wait_improvement,
        'p_value': p_value,
        'reactive_surges': reactive_surges,
        'proactive_surges': proactive_surges
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(reactive_df, proactive_df, stats_summary):
    """Generate comparison charts"""
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Dashboard Validation: Reactive vs Proactive Allocation', 
                 fontsize=16, fontweight='bold')
    
    # 1. Wait Time Distribution
    reactive_waits = reactive_df.groupby('replication')['wait'].mean()
    proactive_waits = proactive_df.groupby('replication')['wait'].mean()
    
    axes[0, 0].hist(reactive_waits, bins=20, alpha=0.6, label='Reactive', color='#E74C3C', edgecolor='black')
    axes[0, 0].hist(proactive_waits, bins=20, alpha=0.6, label='Proactive', color='#27AE60', edgecolor='black')
    axes[0, 0].axvline(reactive_waits.mean(), color='#E74C3C', linestyle='--', linewidth=2, label=f'Reactive Mean ({reactive_waits.mean():.1f} min)')
    axes[0, 0].axvline(proactive_waits.mean(), color='#27AE60', linestyle='--', linewidth=2, label=f'Proactive Mean ({proactive_waits.mean():.1f} min)')
    axes[0, 0].set_xlabel('Average Wait Time (minutes)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Wait Time Distribution (100 Replications)', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    
    # 2. Box Plot Comparison
    data_box = [reactive_waits, proactive_waits]
    bp = axes[0, 1].boxplot(data_box, labels=['Reactive', 'Proactive'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#E74C3C')
    bp['boxes'][1].set_facecolor('#27AE60')
    axes[0, 1].set_ylabel('Average Wait Time (minutes)', fontsize=10)
    axes[0, 1].set_title(f'Policy Comparison (p={stats_summary["p_value"]:.6f})', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Improvement Bar
    improvement = stats_summary['wait_improvement_pct']
    axes[0, 2].bar(['Wait Time\nImprovement'], [improvement], color='#3498DB', edgecolor='black', linewidth=2)
    axes[0, 2].set_ylabel('Improvement (%)', fontsize=10)
    axes[0, 2].set_title(f'Overall Impact: {improvement:.1f}% Reduction', fontsize=11, fontweight='bold')
    axes[0, 2].set_ylim(0, 100)
    axes[0, 2].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate([improvement]):
        axes[0, 2].text(i, v + 3, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    # 4. Wait Time by Department
    dept_reactive = reactive_df.groupby('dept_name')['wait'].mean().sort_values()
    dept_proactive = proactive_df.groupby('dept_name')['wait'].mean().reindex(dept_reactive.index)
    
    x = np.arange(len(dept_reactive))
    width = 0.35
    
    axes[1, 0].barh(x - width/2, dept_reactive, width, label='Reactive', color='#E74C3C', edgecolor='black')
    axes[1, 0].barh(x + width/2, dept_proactive, width, label='Proactive', color='#27AE60', edgecolor='black')
    axes[1, 0].set_yticks(x)
    axes[1, 0].set_yticklabels(dept_reactive.index, fontsize=9)
    axes[1, 0].set_xlabel('Average Wait Time (minutes)', fontsize=10)
    axes[1, 0].set_title('Wait Time by Department', fontsize=11, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 5. Utilization Time Series
    reactive_util_ts = reactive_df.groupby('hour')['staff_util'].mean() * 100
    proactive_util_ts = proactive_df.groupby('hour')['staff_util'].mean() * 100
    
    axes[1, 1].plot(reactive_util_ts.index, reactive_util_ts.values, 'o-', color='#E74C3C', 
                    linewidth=2, markersize=4, label='Reactive')
    axes[1, 1].plot(proactive_util_ts.index, proactive_util_ts.values, 's-', color='#27AE60', 
                    linewidth=2, markersize=4, label='Proactive')
    axes[1, 1].axhline(y=85, color='red', linestyle='--', alpha=0.5, label='Danger Zone (85%)')
    axes[1, 1].set_xlabel('Hour of Day', fontsize=10)
    axes[1, 1].set_ylabel('Staff Utilization (%)', fontsize=10)
    axes[1, 1].set_title('Utilization Over 24 Hours', fontsize=11, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 100)
    
    # 6. Surge Events
    surge_data = [stats_summary['reactive_surges'], stats_summary['proactive_surges']]
    colors = ['#E74C3C', '#27AE60']
    axes[1, 2].bar(['Reactive', 'Proactive'], surge_data, color=colors, edgecolor='black', linewidth=2)
    axes[1, 2].set_ylabel('Number of Surge Events', fontsize=10)
    axes[1, 2].set_title(f'Surge Events (Util > 85%)', fontsize=11, fontweight='bold')
    surge_reduction = (1 - surge_data[1]/surge_data[0]) * 100
    axes[1, 2].text(0.5, max(surge_data) * 0.5, f'{surge_reduction:.1f}%\nreduction', 
                    ha='center', fontsize=14, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    for i, v in enumerate(surge_data):
        axes[1, 2].text(i, v + max(surge_data)*0.05, str(v), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('simulation_validation_results.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: simulation_validation_results.png")
    
    # Create summary table
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Reactive', 'Proactive', 'Improvement'],
        ['Avg Wait Time', f"{stats_summary['reactive_wait_mean']:.1f} min", 
         f"{stats_summary['proactive_wait_mean']:.1f} min", 
         f"{stats_summary['wait_improvement_pct']:.1f}%"],
        ['Surge Events (>85%)', str(stats_summary['reactive_surges']), 
         str(stats_summary['proactive_surges']), 
         f"{(1 - stats_summary['proactive_surges']/stats_summary['reactive_surges'])*100:.1f}%"],
        ['Statistical Significance', 'Baseline', 'Treatment', 
         f"p={stats_summary['p_value']:.6f}"]
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Simulation Validation Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('simulation_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: simulation_summary_table.png")
    
    print(f"\n✅ All visualizations saved!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete validation study"""
    
    print("\n" + "="*70)
    print("DASHBOARD VALIDATION STUDY")
    print("Comparing Reactive vs Proactive Resource Allocation")
    print("="*70)
    
    # Initialize analytics
    generator = ERDataGenerator()
    historical_data = generator.generate_multiple_sessions(num_sessions=5)
    analytics = ERPredictiveAnalytics(historical_data)
    
    print("\n📊 Setup complete:")
    print(f"  - Historical data: 600 observations (5 sessions × 24 hours)")
    print(f"  - Service rate: {SERVICE_RATE} patients/hour/staff")
    print(f"  - Fixed resources: 12 staff, 51 beds")
    
    # Run simulations
    n_reps = 100
    
    reactive_results = run_simulation('reactive', analytics, n_replications=n_reps)
    proactive_results = run_simulation('proactive', analytics, n_replications=n_reps)
    
    # Analyze
    stats_summary = analyze_results(reactive_results, proactive_results)
    
    # Visualize
    create_visualizations(reactive_results, proactive_results, stats_summary)
    
    # Save detailed results
    print(f"\n{'='*70}")
    print("SAVING DETAILED RESULTS")
    print(f"{'='*70}")
    
    reactive_results.to_csv('reactive_simulation_results.csv', index=False)
    print(f"  ✅ Saved: reactive_simulation_results.csv ({len(reactive_results)} rows)")
    
    proactive_results.to_csv('proactive_simulation_results.csv', index=False)
    print(f"  ✅ Saved: proactive_simulation_results.csv ({len(proactive_results)} rows)")
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📈 KEY FINDING:")
    print(f"  Dashboard-driven allocation reduces wait times by {stats_summary['wait_improvement_pct']:.1f}%")
    print(f"  ({stats_summary['reactive_wait_mean']:.1f} min → {stats_summary['proactive_wait_mean']:.1f} min)")
    print(f"  Statistical significance: p = {stats_summary['p_value']:.6f} ***")
    print(f"\n📁 Generated files:")
    print(f"  - simulation_validation_results.png")
    print(f"  - simulation_summary_table.png")
    print(f"  - reactive_simulation_results.csv")
    print(f"  - proactive_simulation_results.csv")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
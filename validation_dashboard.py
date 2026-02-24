"""
COMPREHENSIVE VALIDATION: Game-Based Cost Structure
Uses actual penalty costs from "Friday Night at the ER" game
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
# GAME COST STRUCTURE (from your notes)
# ============================================================================

GAME_COSTS = {
    # Financial penalties (higher is worse)
    'ambulance_diversion': 5000,      # $5,000 per diversion
    'surgery_waiting': 3750,           # $3,750 per patient waiting
    'critical_care_waiting': 3750,     # $3,750 per patient waiting
    'step_down_waiting': 3750,         # $3,750 per patient waiting
    'emergency_waiting': 150,          # $150 per patient waiting
    'extra_staff': 40,                 # $40 per staff hour
    
    # Quality loss penalties (higher is worse)
    'quality_diversion': 200,          # 200 points per diversion
    'quality_waiting': 20,             # 20 points per patient waiting
    'quality_extra_staff': 5,          # 5 points per extra staff hour
}

DEPT_NAMES = {
    'emergency_walkin': 'Emergency Walk-in',
    'emergency_ambulance': 'Emergency Ambulance',
    'surgery': 'Surgical Care',
    'critical_care': 'Critical Care',
    'step_down': 'Step Down'
}

SERVICE_RATE = 2.0  # patients/hour/staff

# ============================================================================
# ALLOCATION POLICIES
# ============================================================================

def reactive_allocation(total_staff=12, total_beds=51):
    """BASELINE: Equal staff split, fixed bed distribution"""
    staff_per_dept = total_staff // len(DEPT_NAMES)
    
    staff_alloc = {dept: staff_per_dept for dept in DEPT_NAMES.keys()}
    
    bed_alloc = {
        'emergency_walkin': 15,
        'emergency_ambulance': 10,
        'surgery': 8,
        'critical_care': 6,
        'step_down': 12
    }
    
    return staff_alloc, bed_alloc


def proactive_allocation(current_patients, forecasts, analytics, total_staff=12, total_beds=51):
    """DASHBOARD-DRIVEN: Priority-weighted M/M/c optimization"""
    
    # Create effective forecasts (current + incoming)
    effective_forecasts = {}
    for dept, fc in forecasts.items():
        current_pts = current_patients[dept]
        new_arrivals = fc['forecast']
        effective_demand = current_pts + new_arrivals
        
        effective_forecasts[dept] = {
            'forecast': effective_demand,
            'lower_bound': fc['lower_bound'] + current_pts,
            'upper_bound': fc['upper_bound'] + current_pts,
            'methods': fc['methods']
        }
    
    allocation_result = analytics.optimize_resource_allocation(
        effective_forecasts, total_staff, total_beds
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
# GAME SCORING FUNCTIONS
# ============================================================================

def calculate_game_costs(hour_results, extra_staff_called):
    """
    Calculate costs using actual game formula
    
    Args:
        hour_results: dict with department metrics for this hour
        extra_staff_called: number of extra staff called this hour
    
    Returns:
        dict with financial_cost, quality_loss, and detailed breakdown
    """
    
    financial_cost = 0
    quality_loss = 0
    breakdown = {
        'ambulance_diversions': 0,
        'high_acuity_waiting': 0,
        'emergency_waiting': 0,
        'extra_staff_cost': 0,
        'quality_diversion': 0,
        'quality_waiting': 0,
        'quality_staff': 0
    }
    
    for dept, metrics in hour_results.items():
        pts_waiting = metrics['patients']
        staff = metrics['staff']
        util = metrics['staff_utilization']
        
        # Ambulance diversion (only for emergency_ambulance when overwhelmed)
        if dept == 'emergency_ambulance' and util > 1.0:
            diversions = int((util - 1.0) * staff * SERVICE_RATE)  # Estimate diversions
            diversions = max(1, diversions) if util > 1.0 else 0
            
            financial_cost += diversions * GAME_COSTS['ambulance_diversion']
            quality_loss += diversions * GAME_COSTS['quality_diversion']
            breakdown['ambulance_diversions'] += diversions
            breakdown['quality_diversion'] += diversions * GAME_COSTS['quality_diversion']
        
        # Waiting penalties
        if pts_waiting > 0:
            if dept in ['surgery', 'critical_care', 'step_down']:
                # High-acuity waiting penalty
                waiting_cost = pts_waiting * GAME_COSTS[f'{dept}_waiting']
                financial_cost += waiting_cost
                breakdown['high_acuity_waiting'] += waiting_cost
            elif dept in ['emergency_walkin', 'emergency_ambulance']:
                # Emergency waiting penalty
                waiting_cost = pts_waiting * GAME_COSTS['emergency_waiting']
                financial_cost += waiting_cost
                breakdown['emergency_waiting'] += waiting_cost
            
            # Quality loss for all waiting
            quality_loss += pts_waiting * GAME_COSTS['quality_waiting']
            breakdown['quality_waiting'] += pts_waiting * GAME_COSTS['quality_waiting']
    
    # Extra staff costs
    if extra_staff_called > 0:
        staff_cost = extra_staff_called * GAME_COSTS['extra_staff']
        financial_cost += staff_cost
        quality_loss += extra_staff_called * GAME_COSTS['quality_extra_staff']
        breakdown['extra_staff_cost'] += staff_cost
        breakdown['quality_staff'] += extra_staff_called * GAME_COSTS['quality_extra_staff']
    
    return {
        'financial_cost': financial_cost,
        'quality_loss': quality_loss,
        'breakdown': breakdown
    }


# ============================================================================
# MONTE CARLO SIMULATION WITH GAME SCORING
# ============================================================================

def run_game_simulation(policy_name, analytics, n_replications=100):
    """Run simulation with game-based cost scoring"""
    
    print(f"\n{'='*70}")
    print(f"RUNNING {policy_name.upper()} POLICY (GAME COSTS)")
    print(f"{'='*70}")
    print(f"Replications: {n_replications}")
    
    # Fitted Poisson rates
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
        
        # Initialize
        current_patients = {dept: 0 for dept in DEPT_NAMES.keys()}
        base_staff = 12
        total_beds = 51
        
        rep_financial_cost = 0
        rep_quality_loss = 0
        rep_extra_staff_total = 0
        rep_diversions = 0
        rep_high_acuity_waiting = 0
        rep_wait_times = []  # Track wait times
        rep_utilizations = []  # Track utilizations
        
        for hour in range(1, 25):
            # Determine phase
            if hour <= 8:
                phase = 'early'
            elif hour <= 16:
                phase = 'mid'
            else:
                phase = 'late'
            
            # Generate arrivals
            new_arrivals = {
                dept: np.random.poisson(lambda_rates[dept][phase])
                for dept in DEPT_NAMES.keys()
            }
            
            # Create forecasts
            forecasts = {
                dept: {
                    'forecast': lambda_rates[dept][phase],
                    'lower_bound': max(0, lambda_rates[dept][phase] - 1.96 * np.sqrt(lambda_rates[dept][phase])),
                    'upper_bound': lambda_rates[dept][phase] + 1.96 * np.sqrt(lambda_rates[dept][phase]),
                    'methods': {'lambda': lambda_rates[dept][phase]}
                }
                for dept in DEPT_NAMES.keys()
            }
            
            # Apply allocation policy
            if policy_name == 'reactive':
                staff_alloc, bed_alloc = reactive_allocation(base_staff, total_beds)
                extra_staff_called = 0
            else:  # proactive
                # Calculate demand per department
                dept_demand = {}
                for dept in DEPT_NAMES.keys():
                    current = current_patients[dept]
                    incoming = new_arrivals.get(dept, 0)
                    dept_demand[dept] = current + incoming
                
                # Check which departments need help (dept-specific surge detection)
                extra_staff_called = 0
                
                # First, allocate base staff optimally
                staff_alloc, bed_alloc = proactive_allocation(
                    current_patients, forecasts, analytics, base_staff, total_beds
                )
                
                # Then check if any critical department is overwhelmed
                for dept in ['emergency_ambulance', 'critical_care', 'surgery']:
                    staff = staff_alloc[dept]
                    demand = dept_demand[dept]
                    
                    if staff > 0:
                        util = demand / (staff * SERVICE_RATE)
                        
                        # Only call extra staff if critical dept is severely overloaded
                        if util > 1.2:  # 120% utilization threshold (more conservative)
                            # Calculate minimal extra staff needed
                            extra_needed = int(np.ceil((demand - staff * SERVICE_RATE * 0.9) / SERVICE_RATE))
                            extra_staff_called += min(extra_needed, 2)  # Max 2 staff per dept
                
                # Cap total extra staff
                extra_staff_called = min(extra_staff_called, 6)  # Max 6 on-call per hour
                
                # If we called extra staff, reallocate with increased capacity
                if extra_staff_called > 0:
                    total_staff = base_staff + extra_staff_called
                    staff_alloc, bed_alloc = proactive_allocation(
                        current_patients, forecasts, analytics, total_staff, total_beds
                    )
            
            # Calculate metrics for current state
            hour_results = {}
            for dept in DEPT_NAMES.keys():
                pts = current_patients[dept]
                staff = staff_alloc[dept]
                
                rho = pts / (staff * SERVICE_RATE) if staff > 0 else 0
                
                # Calculate wait time (M/M/c approximation)
                if rho >= 1.0 or staff == 0:
                    wait = 999  # Unstable
                else:
                    wait = (rho / (staff * SERVICE_RATE * (1 - rho))) * 15
                
                hour_results[dept] = {
                    'patients': pts,
                    'staff': staff,
                    'staff_utilization': rho,
                    'wait_time': wait
                }
                
                # Track wait times and utilization (only stable states)
                if wait < 999:
                    rep_wait_times.append(wait)
                    rep_utilizations.append(rho)
            
            # Calculate costs for this hour
            costs = calculate_game_costs(hour_results, extra_staff_called)
            
            rep_financial_cost += costs['financial_cost']
            rep_quality_loss += costs['quality_loss']
            rep_extra_staff_total += extra_staff_called
            rep_diversions += costs['breakdown']['ambulance_diversions']
            rep_high_acuity_waiting += costs['breakdown']['high_acuity_waiting']
            
            # Update patient counts (discharge + arrivals)
            for dept in DEPT_NAMES.keys():
                discharged = int(current_patients[dept] * 0.4)
                current_patients[dept] = max(0, current_patients[dept] - discharged)
                current_patients[dept] += new_arrivals[dept]
        
        # Store replication results
        all_results.append({
            'replication': rep + 1,
            'financial_cost': rep_financial_cost,
            'quality_loss': rep_quality_loss,
            'extra_staff_hours': rep_extra_staff_total,
            'ambulance_diversions': rep_diversions,
            'high_acuity_waiting_cost': rep_high_acuity_waiting,
            'avg_wait_time': np.mean(rep_wait_times) if rep_wait_times else 999,
            'avg_utilization': np.mean(rep_utilizations) if rep_utilizations else 0
        })
    
    print(f"  ✅ Completed all {n_replications} replications")
    
    return pd.DataFrame(all_results)


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_game_results(reactive_df, proactive_df):
    """Compare policies using game costs"""
    
    print(f"\n{'='*70}")
    print("GAME-BASED COST ANALYSIS")
    print(f"{'='*70}")
    
    # Financial Cost
    print("\n💰 FINANCIAL COST:")
    print(f"  Reactive:  ${reactive_df['financial_cost'].mean():,.0f} (±${reactive_df['financial_cost'].std():,.0f})")
    print(f"  Proactive: ${proactive_df['financial_cost'].mean():,.0f} (±${proactive_df['financial_cost'].std():,.0f})")
    
    financial_improvement = (reactive_df['financial_cost'].mean() - proactive_df['financial_cost'].mean()) / reactive_df['financial_cost'].mean() * 100
    financial_savings = reactive_df['financial_cost'].mean() - proactive_df['financial_cost'].mean()
    
    print(f"  Improvement: {financial_improvement:.1f}%")
    print(f"  Savings: ${financial_savings:,.0f} per 24-hour shift")
    print(f"  Annual Savings: ${financial_savings * 365:,.0f}")
    
    t_stat, p_value = stats.ttest_ind(reactive_df['financial_cost'], proactive_df['financial_cost'])
    print(f"  T-test: t={t_stat:.2f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # Wait Time Analysis
    print("\n⏱️ WAIT TIME:")
    reactive_wait = reactive_df['avg_wait_time'].mean()
    proactive_wait = proactive_df['avg_wait_time'].mean()
    wait_improvement = (reactive_wait - proactive_wait) / reactive_wait * 100
    
    print(f"  Reactive:  {reactive_wait:.2f} min (±{reactive_df['avg_wait_time'].std():.2f})")
    print(f"  Proactive: {proactive_wait:.2f} min (±{proactive_df['avg_wait_time'].std():.2f})")
    print(f"  Improvement: {wait_improvement:.1f}%")
    
    # Utilization Analysis
    print("\n📊 UTILIZATION:")
    reactive_util = reactive_df['avg_utilization'].mean() * 100
    proactive_util = proactive_df['avg_utilization'].mean() * 100
    
    print(f"  Reactive:  {reactive_util:.1f}%")
    print(f"  Proactive: {proactive_util:.1f}%")
    print(f"  Change: {proactive_util - reactive_util:+.1f} percentage points")
    
    # Quality Loss
    print("\n⭐ QUALITY LOSS:")
    print(f"  Reactive:  {reactive_df['quality_loss'].mean():,.0f} points (±{reactive_df['quality_loss'].std():,.0f})")
    print(f"  Proactive: {proactive_df['quality_loss'].mean():,.0f} points (±{proactive_df['quality_loss'].std():,.0f})")
    
    quality_improvement = (reactive_df['quality_loss'].mean() - proactive_df['quality_loss'].mean()) / reactive_df['quality_loss'].mean() * 100
    print(f"  Improvement: {quality_improvement:.1f}%")
    
    # Ambulance Diversions
    print("\n🚑 AMBULANCE DIVERSIONS:")
    print(f"  Reactive:  {reactive_df['ambulance_diversions'].sum()} total diversions")
    print(f"  Proactive: {proactive_df['ambulance_diversions'].sum()} total diversions")
    print(f"  Reduction: {reactive_df['ambulance_diversions'].sum() - proactive_df['ambulance_diversions'].sum()} diversions prevented")
    print(f"  Savings from diversions alone: ${(reactive_df['ambulance_diversions'].sum() - proactive_df['ambulance_diversions'].sum()) * GAME_COSTS['ambulance_diversion']:,.0f}")
    
    # Extra Staff Usage
    print("\n👥 EXTRA STAFF (ON-CALL):")
    print(f"  Reactive:  {reactive_df['extra_staff_hours'].mean():.1f} hours/shift")
    print(f"  Proactive: {proactive_df['extra_staff_hours'].mean():.1f} hours/shift")
    print(f"  Difference: {proactive_df['extra_staff_hours'].mean() - reactive_df['extra_staff_hours'].mean():.1f} more hours")
    print(f"  Cost: ${(proactive_df['extra_staff_hours'].mean() - reactive_df['extra_staff_hours'].mean()) * GAME_COSTS['extra_staff']:,.0f} per shift")
    
    # ROI Calculation
    print("\n📊 RETURN ON INVESTMENT:")
    extra_staff_cost = (proactive_df['extra_staff_hours'].mean() - reactive_df['extra_staff_hours'].mean()) * GAME_COSTS['extra_staff']
    total_savings = financial_savings
    roi = (total_savings / extra_staff_cost - 1) * 100 if extra_staff_cost > 0 else float('inf')
    print(f"  Investment: ${extra_staff_cost:,.0f} (extra staff)")
    print(f"  Return: ${total_savings:,.0f} (cost reduction)")
    print(f"  ROI: {roi:.0f}%")
    
    return {
        'reactive_cost_mean': reactive_df['financial_cost'].mean(),
        'proactive_cost_mean': proactive_df['financial_cost'].mean(),
        'cost_improvement_pct': financial_improvement,
        'savings_per_shift': financial_savings,
        'p_value': p_value,
        'reactive_quality': reactive_df['quality_loss'].mean(),
        'proactive_quality': proactive_df['quality_loss'].mean(),
        'quality_improvement_pct': quality_improvement,
        'diversions_prevented': reactive_df['ambulance_diversions'].sum() - proactive_df['ambulance_diversions'].sum(),
        'reactive_wait': reactive_wait,
        'proactive_wait': proactive_wait,
        'wait_improvement_pct': wait_improvement,
        'reactive_util': reactive_util,
        'proactive_util': proactive_util
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_game_visualizations(reactive_df, proactive_df, stats_summary):
    """Generate comparison charts with game costs"""
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comprehensive Dashboard Validation: Game Costs + Performance Metrics', fontsize=16, fontweight='bold')
    
    # 1. Wait Time Distribution
    axes[0, 0].hist(reactive_df['avg_wait_time'], bins=20, alpha=0.6, label='Reactive', color='#E74C3C', edgecolor='black')
    axes[0, 0].hist(proactive_df['avg_wait_time'], bins=20, alpha=0.6, label='Proactive', color='#27AE60', edgecolor='black')
    axes[0, 0].axvline(reactive_df['avg_wait_time'].mean(), color='#E74C3C', linestyle='--', linewidth=2)
    axes[0, 0].axvline(proactive_df['avg_wait_time'].mean(), color='#27AE60', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Average Wait Time (minutes)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title(f'Wait Time: {stats_summary["wait_improvement_pct"]:.1f}% Improvement', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    
    # 2. Financial Cost Box Plot
    data_box = [reactive_df['financial_cost'], proactive_df['financial_cost']]
    bp = axes[0, 1].boxplot(data_box, tick_labels=['Reactive', 'Proactive'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#E74C3C')
    bp['boxes'][1].set_facecolor('#27AE60')
    axes[0, 1].set_ylabel('Financial Cost ($)', fontsize=10)
    axes[0, 1].set_title(f'Cost: {stats_summary["cost_improvement_pct"]:.1f}% Improvement (p={stats_summary["p_value"]:.4f})', fontsize=11, fontweight='bold')
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    
    # 3. Dual Improvement Bar
    improvements = {
        'Wait\nTime': stats_summary['wait_improvement_pct'],
        'Financial\nCost': stats_summary['cost_improvement_pct']
    }
    colors = ['#27AE60' if v > 0 else '#E74C3C' for v in improvements.values()]
    axes[0, 2].bar(improvements.keys(), improvements.values(), color=colors, edgecolor='black', linewidth=2)
    axes[0, 2].set_ylabel('Improvement (%)', fontsize=10)
    axes[0, 2].set_title(f'Overall Impact', fontsize=11, fontweight='bold')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    for i, (k, v) in enumerate(improvements.items()):
        axes[0, 2].text(i, v + (2 if v > 0 else -2), f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # 4. Quality Loss
    axes[1, 0].bar(['Reactive', 'Proactive'], 
                   [stats_summary['reactive_quality'], stats_summary['proactive_quality']],
                   color=['#E74C3C', '#27AE60'], edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Quality Loss (points)', fontsize=10)
    axes[1, 0].set_title(f'Quality Loss: {stats_summary["quality_improvement_pct"]:.1f}% Reduction', fontsize=11, fontweight='bold')
    
    # 5. Savings Breakdown
    savings_data = {
        'Cost\nReduction': stats_summary['savings_per_shift'],
        'Diversion\nSavings': stats_summary['diversions_prevented'] * GAME_COSTS['ambulance_diversion']
    }
    axes[1, 1].bar(savings_data.keys(), savings_data.values(), color='#3498DB', edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Savings per Shift ($)', fontsize=10)
    axes[1, 1].set_title('Savings Breakdown', fontsize=11, fontweight='bold')
    axes[1, 1].ticklabel_format(style='plain', axis='y')
    for i, (k, v) in enumerate(savings_data.items()):
        axes[1, 1].text(i, v + max(savings_data.values())*0.05, f'${v:,.0f}', ha='center', fontsize=10, fontweight='bold')
    
    # 6. ROI Summary
    extra_cost = (proactive_df['extra_staff_hours'].mean() - reactive_df['extra_staff_hours'].mean()) * GAME_COSTS['extra_staff']
    roi = (stats_summary['savings_per_shift'] / extra_cost - 1) * 100 if extra_cost > 0 else 0
    
    roi_data = {'Investment': extra_cost, 'Return': stats_summary['savings_per_shift']}
    axes[1, 2].bar(roi_data.keys(), roi_data.values(), color=['#E74C3C', '#27AE60'], edgecolor='black', linewidth=2)
    axes[1, 2].set_ylabel('Amount ($)', fontsize=10)
    axes[1, 2].set_title(f'ROI: {roi:.0f}%', fontsize=11, fontweight='bold')
    axes[1, 2].ticklabel_format(style='plain', axis='y')
    for i, (k, v) in enumerate(roi_data.items()):
        axes[1, 2].text(i, v + max(roi_data.values())*0.05, f'${v:,.0f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('game_cost_validation_results.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: game_cost_validation_results.png")
    
    # Summary table
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Reactive', 'Proactive', 'Improvement'],
        ['Wait Time', f"{stats_summary['reactive_wait']:.1f} min", 
         f"{stats_summary['proactive_wait']:.1f} min", 
         f"{stats_summary['wait_improvement_pct']:.1f}%"],
        ['Utilization', f"{stats_summary['reactive_util']:.1f}%", 
         f"{stats_summary['proactive_util']:.1f}%", 
         f"{stats_summary['proactive_util'] - stats_summary['reactive_util']:+.1f}pp"],
        ['Financial Cost', f"${stats_summary['reactive_cost_mean']:,.0f}", 
         f"${stats_summary['proactive_cost_mean']:,.0f}", 
         f"{stats_summary['cost_improvement_pct']:.1f}%"],
        ['Quality Loss', f"{stats_summary['reactive_quality']:,.0f} pts", 
         f"{stats_summary['proactive_quality']:,.0f} pts", 
         f"{stats_summary['quality_improvement_pct']:.1f}%"],
        ['Diversions Prevented', '—', '—', 
         f"{stats_summary['diversions_prevented']} events"],
        ['Savings per Shift', '—', '—', 
         f"${stats_summary['savings_per_shift']:,.0f}"],
        ['Annual Savings', '—', '—', 
         f"${stats_summary['savings_per_shift'] * 365:,.0f}"],
        ['Statistical Sig.', 'Baseline', 'Treatment', 
         f"p={stats_summary['p_value']:.6f}"]
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight improvements
    for row in [1, 2, 3, 4, 5]:
        table[(row, 3)].set_facecolor('#D5F4E6')
        table[(row, 3)].set_text_props(weight='bold', color='#27AE60')
    
    plt.title('Comprehensive Validation Summary (Wait Time + Financial + Quality)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('game_cost_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: game_cost_summary_table.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("GAME-BASED COST VALIDATION")
    print("Using Actual Friday Night at the ER Penalties")
    print("="*70)
    
    # Initialize
    generator = ERDataGenerator()
    historical_data = generator.generate_multiple_sessions(num_sessions=5)
    analytics = ERPredictiveAnalytics(historical_data)
    
    print("\n📊 Game Cost Structure:")
    print(f"  - Ambulance Diversion: ${GAME_COSTS['ambulance_diversion']:,}")
    print(f"  - High-Acuity Waiting: ${GAME_COSTS['surgery_waiting']:,} per patient")
    print(f"  - Emergency Waiting: ${GAME_COSTS['emergency_waiting']:,} per patient")
    print(f"  - Extra Staff: ${GAME_COSTS['extra_staff']:,} per hour")
    
    # Run simulations
    n_reps = 100
    
    reactive_results = run_game_simulation('reactive', analytics, n_replications=n_reps)
    proactive_results = run_game_simulation('proactive', analytics, n_replications=n_reps)
    
    # Analyze
    stats_summary = analyze_game_results(reactive_results, proactive_results)
    
    # Visualize
    create_game_visualizations(reactive_results, proactive_results, stats_summary)
    
    # Save CSVs
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    reactive_results.to_csv('reactive_game_costs.csv', index=False)
    print(f"  ✅ Saved: reactive_game_costs.csv")
    
    proactive_results.to_csv('proactive_game_costs.csv', index=False)
    print(f"  ✅ Saved: proactive_game_costs.csv")
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n🎯 KEY FINDINGS:")
    print(f"  Wait Time Reduction: {stats_summary['wait_improvement_pct']:.1f}%")
    print(f"  Financial Cost Reduction: {stats_summary['cost_improvement_pct']:.1f}%")
    print(f"  Savings per 24-hour shift: ${stats_summary['savings_per_shift']:,.0f}")
    print(f"  Annual savings: ${stats_summary['savings_per_shift'] * 365:,.0f}")
    print(f"  Quality improvement: {stats_summary['quality_improvement_pct']:.1f}%")
    print(f"  Ambulance diversions prevented: {stats_summary['diversions_prevented']}")
    print(f"  Statistical significance: p = {stats_summary['p_value']:.6f}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
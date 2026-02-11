"""
Healthcare ER Patient Flow - Enhanced Data Generator
Generates data using Poisson distribution (theoretically correct for count data)
Includes patient discharge/treatment simulation
Exports historical data for dashboard use
"""

import numpy as np
import pandas as pd

class ERDataGenerator:
    """
    Generate realistic ER patient arrival data using Poisson distribution
    
    Poisson is theoretically correct for:
    - Discrete count data (# of patients)
    - Independent arrivals
    - Constant average rate per period
    """
    
    def __init__(self):
        # Actual gameplay data (23 rounds) - serves as validation
        self.actual_data = {
            'emergency_walkin': [2,4,3,8,4,5,5,7,5,4,4,5,6,4,6,2,2,1,7,1,7,4,2],
            'emergency_ambulance': [0,1,1,2,0,2,0,0,1,2,1,3,2,2,2,3,1,1,0,1,0,2,1],
            'surgery': [3,1,1,0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'critical_care': [1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'step_down': [1,2,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1]
        }
        
        # Fit Poisson models to actual data
        self.poisson_params = self._fit_poisson_models()
        
        # Patient discharge/treatment parameters
        self.discharge_rates = {
            'emergency_walkin': 0.6,      # 60% discharged per round
            'emergency_ambulance': 0.5,   # 50% discharged per round
            'surgery': 0.3,               # 30% transferred to step-down
            'critical_care': 0.4,         # 40% transferred to step-down
            'step_down': 0.7              # 70% discharged home
        }
    
    def _fit_poisson_models(self):
        """
        Fit Poisson distribution parameters (lambda) to actual data
        Separate parameters for early/mid/late game phases
        """
        params = {}
        
        for dept, values in self.actual_data.items():
            # Split into phases
            early_data = values[:8]    # Rounds 1-8
            mid_data = values[8:16]    # Rounds 9-16
            late_data = values[16:]    # Rounds 17-23
            
            params[dept] = {
                'early_lambda': np.mean(early_data),
                'mid_lambda': np.mean(mid_data),
                'late_lambda': np.mean(late_data),
                'overall_lambda': np.mean(values)
            }
        
        return params
    
    def _get_lambda_for_round(self, dept, round_num):
        """Get appropriate lambda parameter for department and round"""
        params = self.poisson_params[dept]
        
        if round_num <= 8:
            return params['early_lambda']
        elif round_num <= 16:
            return params['mid_lambda']
        else:
            return params['late_lambda']
    
    def generate_round_arrivals(self, dept, round_num):
        """
        Generate arrivals using Poisson distribution
        
        Args:
            dept: Department name
            round_num: Round number (1-23)
        
        Returns:
            Number of arriving patients (Poisson-distributed)
        """
        lambda_param = self._get_lambda_for_round(dept, round_num)
        
        # Generate from Poisson distribution
        arrivals = np.random.poisson(lambda_param)
        
        return arrivals
    
    def simulate_patient_discharge(self, current_patients, dept):
        """
        Simulate patients being treated/discharged
        
        Args:
            current_patients: Current number of patients in department
            dept: Department name
        
        Returns:
            Number of patients remaining after treatment/discharge
        """
        if current_patients == 0:
            return 0
        
        discharge_rate = self.discharge_rates.get(dept, 0.5)
        
        # Binomial distribution: each patient has probability of being discharged
        discharged = np.random.binomial(current_patients, discharge_rate)
        
        remaining = current_patients - discharged
        
        return remaining
    
    def generate_session(self, num_rounds=23, session_id=1):
        """
        Generate a complete gameplay session using Poisson distribution
        
        Returns:
            DataFrame with round-by-round arrivals
        """
        session_data = {
            'round': [],
            'session_id': [],
            'emergency_walkin': [],
            'emergency_ambulance': [],
            'surgery': [],
            'critical_care': [],
            'step_down': []
        }
        
        for round_num in range(1, num_rounds + 1):
            session_data['round'].append(round_num)
            session_data['session_id'].append(session_id)
            
            for dept in ['emergency_walkin', 'emergency_ambulance', 'surgery', 
                        'critical_care', 'step_down']:
                arrivals = self.generate_round_arrivals(dept, round_num)
                session_data[dept].append(arrivals)
        
        return pd.DataFrame(session_data)
    
    def generate_multiple_sessions(self, num_sessions=5, num_rounds=23):
        """
        Generate multiple gameplay sessions with Poisson variability
        
        Returns:
            DataFrame with all sessions
        """
        all_sessions = []
        
        # First session is actual data
        actual_df = pd.DataFrame(self.actual_data)
        actual_df.insert(0, 'round', list(range(1, len(actual_df) + 1)))
        actual_df.insert(1, 'session_id', 0)
        all_sessions.append(actual_df)
        
        # Generate additional sessions using fitted Poisson models
        for session_id in range(1, num_sessions):
            session_df = self.generate_session(num_rounds, session_id)
            all_sessions.append(session_df)
        
        combined_df = pd.concat(all_sessions, ignore_index=True)
        
        return combined_df
    
    def generate_real_time_data(self, current_round, historical_df):
        """
        Generate real-time arrivals for current round
        
        Args:
            current_round: Current round number
            historical_df: Historical data (not used in Poisson approach)
        
        Returns:
            Dict of arrivals by department
        """
        current_data = {}
        
        for dept in ['emergency_walkin', 'emergency_ambulance', 'surgery', 
                    'critical_care', 'step_down']:
            arrivals = self.generate_round_arrivals(dept, current_round)
            current_data[dept] = arrivals
        
        return current_data
    
    def export_to_csv(self, df, filename='er_historical_data.csv'):
        """Export generated Poisson data to CSV"""
        df.to_csv(filename, index=False)
        print(f"✓ Poisson-based historical data exported to {filename}")
        print(f"  - {len(df)} total records")
        print(f"  - {len(df['session_id'].unique())} sessions")
        print(f"  - Fitted using Poisson distribution (theoretically correct for count data)")
        return filename
    
    def get_poisson_parameters(self):
        """Return fitted Poisson parameters for reporting"""
        return self.poisson_params


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ER DATA GENERATOR - POISSON DISTRIBUTION")
    print("="*70)
    
    # Generate data
    generator = ERDataGenerator()
    
    # Show fitted parameters
    print("\n📊 Fitted Poisson Parameters (λ):")
    print("-" * 70)
    for dept, params in generator.poisson_params.items():
        print(f"\n{dept.replace('_', ' ').title()}:")
        print(f"  Early Game (Rounds 1-8):   λ = {params['early_lambda']:.2f}")
        print(f"  Mid Game (Rounds 9-16):    λ = {params['mid_lambda']:.2f}")
        print(f"  Late Game (Rounds 17-23):  λ = {params['late_lambda']:.2f}")
    
    # Generate 5 sessions (1 actual + 4 Poisson-generated)
    print("\n\n📦 Generating Historical Data...")
    historical_data = generator.generate_multiple_sessions(num_sessions=5)
    
    print(f"\n✓ Generated {len(historical_data)} records")
    print(f"✓ Sessions: {list(historical_data['session_id'].unique())}")
    
    print("\n📋 Sample Data (First 10 rows):")
    print(historical_data.head(10).to_string(index=False))
    
    print("\n📊 Statistical Summary:")
    print(historical_data[['emergency_walkin', 'emergency_ambulance', 'surgery', 
                          'critical_care', 'step_down']].describe().round(2))
    
    # Export
    print("\n\n💾 Exporting Data...")
    generator.export_to_csv(historical_data, 'er_historical_data.csv')
    
    # Test discharge simulation
    print("\n\n🏥 Testing Patient Discharge Simulation:")
    print("-" * 70)
    test_patients = {'emergency_walkin': 10, 'surgery': 5, 'step_down': 8}
    print("\nInitial Patients:")
    for dept, count in test_patients.items():
        print(f"  {dept}: {count} patients")
    
    print("\nAfter Treatment/Discharge:")
    for dept, count in test_patients.items():
        remaining = generator.simulate_patient_discharge(count, dept)
        discharged = count - remaining
        rate = generator.discharge_rates[dept]
        print(f"  {dept}: {remaining} remain ({discharged} discharged, rate={rate:.0%})")
    
    print("\n" + "="*70)
    print("✓ Data generation complete!")
    print("="*70 + "\n")
"""
Healthcare ER Patient Flow - Discrete Event Simulation
Based on operational parameters and causal relationships
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ERSimulationModel:
    """
    Discrete Event Simulation for ER patient flow
    Based on operational parameters, not just statistical fitting
    """
    
    def __init__(self):
        # Model PARAMETERS (from game mechanics)
        self.arrival_rates = {
            'emergency_walkin': {
                'early': 5.0,    # λ (lambda) - avg arrivals per round
                'mid': 4.5,
                'late': 3.5
            },
            'emergency_ambulance': {
                'early': 1.2,
                'mid': 1.5,
                'late': 0.8
            }
        }
        
        # Process dependencies
        self.surgery_to_stepdown_prob = 0.6  # 60% of surgery → step-down
        self.critical_to_stepdown_prob = 0.4
        
        # Surgery/Critical care parameters
        self.surgery_early_rate = 1.5
        self.critical_early_rate = 0.6
        
    def _get_phase(self, round_num):
        """Determine game phase based on round number"""
        if round_num < 8:
            return 'early'
        elif round_num < 16:
            return 'mid'
        else:
            return 'late'
    
    def _generate_surgery_arrivals(self, round_num, current_state):
        """Surgery arrivals - concentrated in early game"""
        if round_num < 8:
            # Early game: active surgery arrivals
            base_rate = self.surgery_early_rate
            # Reduce if capacity is stressed
            if current_state.get('capacity_stress', 0) > 0.7:
                base_rate *= 0.5
            return np.random.poisson(base_rate)
        else:
            # After round 8: very rare
            return np.random.poisson(0.1)
    
    def _generate_critical_arrivals(self, round_num, current_state):
        """Critical care arrivals - concentrated in early game"""
        if round_num < 8:
            # Early game: active critical arrivals
            base_rate = self.critical_early_rate
            # Reduce if capacity is stressed
            if current_state.get('capacity_stress', 0) > 0.7:
                base_rate *= 0.6
            return np.random.poisson(base_rate)
        else:
            # After round 8: very rare
            return np.random.poisson(0.05)
    
    def _generate_stepdown_arrivals(self, round_num, prev_surgery, prev_critical):
        """Step-down arrivals come from previous specialty treatments"""
        # Downstream flows from surgery and critical care
        expected = (prev_surgery * self.surgery_to_stepdown_prob + 
                   prev_critical * self.critical_to_stepdown_prob)
        
        # Add some random walk-ins (front-loaded)
        if round_num < 10:
            expected += np.random.poisson(0.8)
        elif round_num >= 19:
            # Occasional late arrivals
            expected += np.random.poisson(0.3)
        
        return np.random.poisson(expected) if expected > 0 else 0
    
    def simulate_round(self, round_num, current_state):
        """
        Generate arrivals based on:
        1. Time-dependent arrival rates (Poisson process)
        2. State-dependent factors (capacity, previous round outcomes)
        3. Downstream patient flows (surgery → step-down)
        """
        # Primary arrivals (Poisson distributed)
        phase = self._get_phase(round_num)
        emergency_walkin = np.random.poisson(
            self.arrival_rates['emergency_walkin'][phase]
        )
        emergency_ambulance = np.random.poisson(
            self.arrival_rates['emergency_ambulance'][phase]
        )
        
        # Specialty arrivals (dependent on game phase and capacity)
        surgery = self._generate_surgery_arrivals(round_num, current_state)
        critical_care = self._generate_critical_arrivals(round_num, current_state)
        
        # Downstream flows (from previous treatments)
        step_down = self._generate_stepdown_arrivals(
            round_num, 
            current_state.get('previous_surgery', 0),
            current_state.get('previous_critical', 0)
        )
        
        return {
            'emergency_walkin': emergency_walkin,
            'emergency_ambulance': emergency_ambulance,
            'surgery': surgery,
            'critical_care': critical_care,
            'step_down': step_down
        }
    
    def simulate_session(self, num_rounds=23, session_id=1):
        """Simulate a complete gameplay session"""
        session_data = {
            'round': [],
            'session_id': [],
            'emergency_walkin': [],
            'emergency_ambulance': [],
            'surgery': [],
            'critical_care': [],
            'step_down': []
        }
        
        # Initialize state
        current_state = {
            'previous_surgery': 0,
            'previous_critical': 0,
            'capacity_stress': 0.0
        }
        
        for round_num in range(num_rounds):
            # Simulate this round
            arrivals = self.simulate_round(round_num, current_state)
            
            # Record data
            session_data['round'].append(round_num + 1)
            session_data['session_id'].append(session_id)
            session_data['emergency_walkin'].append(arrivals['emergency_walkin'])
            session_data['emergency_ambulance'].append(arrivals['emergency_ambulance'])
            session_data['surgery'].append(arrivals['surgery'])
            session_data['critical_care'].append(arrivals['critical_care'])
            session_data['step_down'].append(arrivals['step_down'])
            
            # Update state for next round
            current_state['previous_surgery'] = arrivals['surgery']
            current_state['previous_critical'] = arrivals['critical_care']
            
            # Simple capacity stress calculation (total arrivals / 20)
            total_arrivals = sum(arrivals.values())
            current_state['capacity_stress'] = min(1.0, total_arrivals / 20.0)
        
        return pd.DataFrame(session_data)
    
    def simulate_multiple_sessions(self, num_sessions=5, num_rounds=23):
        """Generate multiple simulation runs"""
        all_sessions = []
        
        for session_id in range(num_sessions):
            session_df = self.simulate_session(num_rounds, session_id)
            all_sessions.append(session_df)
        
        return pd.concat(all_sessions, ignore_index=True)
    
    def export_to_csv(self, df, filename='er_simulation_data.csv'):
        """Export simulated data to CSV"""
        df.to_csv(filename, index=False)
        print(f"Simulation data exported to {filename}")
        return filename


if __name__ == "__main__":
    # Create simulation model
    simulator = ERSimulationModel()
    
    # Generate 5 simulation runs
    print("Running Discrete Event Simulation...")
    simulation_data = simulator.simulate_multiple_sessions(num_sessions=5, num_rounds=23)
    
    print("\nSimulation Results:")
    print(f"Total records: {len(simulation_data)}")
    print(f"Sessions: {simulation_data['session_id'].unique()}")
    print("\nSample data:")
    print(simulation_data.head(10))
    print("\nStatistics by department:")
    print(simulation_data[['emergency_walkin', 'emergency_ambulance', 'surgery', 'critical_care', 'step_down']].describe())
    
    # Export
    simulator.export_to_csv(simulation_data, 'er_simulation_data.csv')
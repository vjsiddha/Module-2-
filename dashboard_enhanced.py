"""
Enhanced ER Patient Flow Dashboard
- User-adjustable parameters
- Game state editing
- Reasoning-backed recommendations
- Integrated Poisson forecasting
- Combined staff + bed allocation
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from data_generator import ERDataGenerator
try:
    from predictive_analytics_enhanced import ERPredictiveAnalytics
except ImportError:
    from predictive_analytics import ERPredictiveAnalytics

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Generate historical data
generator = ERDataGenerator()
historical_data = generator.generate_multiple_sessions(num_sessions=5)

# Initialize analytics
analytics = ERPredictiveAnalytics(historical_data)

# Department names
DEPT_NAMES = {
    'emergency_walkin': 'Emergency Walk-in',
    'emergency_ambulance': 'Emergency Ambulance',
    'surgery': 'Surgical Care',
    'critical_care': 'Critical Care',
    'step_down': 'Step Down'
}

# Department colors
DEPT_COLORS = {
    'emergency_walkin': '#FF6B6B',
    'emergency_ambulance': '#EE5A6F',
    'surgery': '#4ECDC4',
    'critical_care': '#FFA07A',
    'step_down': '#95E1D3'
}

# Initial game state
game_state = {
    'current_hour': 1,  # Hour 1-24 instead of rounds
    'total_staff': 12,
    'total_beds': 51,
    'current_patients': {dept: 0 for dept in DEPT_NAMES.keys()},
    'staff_allocation': {},
    'bed_allocation': {},
    'total_treated': 0
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏥 ER Command Center - Enhanced Decision Support", 
                style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '5px'}),
        html.P("Poisson-Based Forecasting | Reasoning-Backed Recommendations | Interactive Controls",
               style={'textAlign': 'center', 'color': '#7F8C8D', 'fontSize': '14px'})
    ], style={'backgroundColor': '#ECF0F1', 'padding': '15px', 'marginBottom': '15px'}),
    
    # === PARAMETER CONTROL PANEL ===
    html.Div([
        html.H3("🎛️ Game Parameters & Controls", style={'color': '#2C3E50', 'marginBottom': '15px'}),
        
        html.Div([
            # Hour control
            html.Div([
                html.Label("Current Hour:", style={'fontWeight': 'bold'}),
                dcc.Input(id='current-hour', type='number', value=1, min=1, max=24,
                         style={'width': '70px', 'marginLeft': '10px'}),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),
            
            # Total staff control
            html.Div([
                html.Label("Total Staff Available:", style={'fontWeight': 'bold'}),
                dcc.Input(id='total-staff', type='number', value=12, min=5, max=30,
                         style={'width': '70px', 'marginLeft': '10px'}),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),
            
            # Total beds control
            html.Div([
                html.Label("Total Beds Available:", style={'fontWeight': 'bold'}),
                dcc.Input(id='total-beds', type='number', value=51, min=20, max=100,
                         style={'width': '70px', 'marginLeft': '10px'}),
            ], style={'display': 'inline-block'}),
        ]),
        
        html.Div([
            html.Button('🔄 Update & Recalculate', id='update-btn', n_clicks=0,
                       style={'backgroundColor': '#3498DB', 'color': 'white', 'border': 'none',
                              'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px',
                              'marginTop': '10px', 'marginRight': '10px'}),
            html.Button('▶️ Simulate Next Hour', id='simulate-btn', n_clicks=0,
                       style={'backgroundColor': '#2ECC71', 'color': 'white', 'border': 'none',
                              'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px',
                              'marginTop': '10px'}),
        ]),
        
        html.Div(id='parameter-status', style={'marginTop': '10px', 'color': '#27AE60', 'fontWeight': 'bold'})
    ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '15px',
             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # === EDIT GAME STATE ===
    html.Div([
        html.H3("✏️ Edit Current Game State", style={'color': '#2C3E50', 'marginBottom': '15px'}),
        html.P("Manually set current patient counts to match your actual gameplay", 
               style={'fontSize': '12px', 'color': '#7F8C8D', 'marginBottom': '10px'}),
        
        dash_table.DataTable(
            id='game-state-table',
            columns=[
                {'name': 'Department', 'id': 'department', 'editable': False},
                {'name': 'Current Patients', 'id': 'current_patients', 'editable': True, 'type': 'numeric'},
            ],
            data=[],
            editable=True,
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '13px'},
            style_header={'backgroundColor': '#E74C3C', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'column_id': 'department'}, 'fontWeight': 'bold', 'backgroundColor': '#F8F9FA'}
            ]
        ),
        
        html.Button('💾 Save Game State', id='save-state-btn', n_clicks=0,
                   style={'backgroundColor': '#E74C3C', 'color': 'white', 'border': 'none',
                          'padding': '10px 20px', 'cursor': 'pointer', 'borderRadius': '5px',
                          'marginTop': '10px'}),
        html.Div(id='save-state-status', style={'display': 'inline-block', 'marginLeft': '15px',
                                                'color': '#27AE60', 'fontWeight': 'bold'})
    ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '15px',
             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Alerts
    html.Div(id='alert-panel', style={'marginBottom': '15px'}),
    
    # Heat Map - Quick Status View
    html.Div([
        html.H3("🔥 Department Utilization Heat Map", style={'color': '#2C3E50', 'marginBottom': '15px'}),
        html.P("Real-time capacity utilization across all departments",
               style={'fontSize': '11px', 'color': '#7F8C8D', 'marginBottom': '10px', 'fontStyle': 'italic'}),
        dcc.Graph(id='heatmap-chart')
    ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '15px',
             'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # === MAIN DASHBOARD: FORECAST & RECOMMENDATIONS ===
    html.Div([
        # Left: Forecast
        html.Div([
            html.Div([
                html.H3("🔮 Poisson Distribution Forecast", style={'color': '#2C3E50', 'marginBottom': '15px'}),
                html.P("Next 4 hours prediction based on fitted Poisson models (theoretically correct for count data)",
                      style={'fontSize': '11px', 'color': '#7F8C8D', 'marginBottom': '10px', 'fontStyle': 'italic'}),
                dcc.Graph(id='forecast-chart')
            ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '15px',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Forecast Details with Reasoning
            html.Div([
                html.H3("📊 Forecast Details & Statistical Reasoning", style={'color': '#2C3E50', 'marginBottom': '15px'}),
                html.Div(id='forecast-reasoning')
            ], style={'backgroundColor': 'white', 'padding': '20px',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right: Recommendations
        html.Div([
            html.Div([
                html.H3("💡 Optimized Resource Allocation", style={'color': '#2C3E50', 'marginBottom': '15px'}),
                html.P("Staff + Bed allocation based on M/M/c queueing theory & priority weighting",
                      style={'fontSize': '11px', 'color': '#7F8C8D', 'marginBottom': '10px', 'fontStyle': 'italic'}),
                html.Div(id='resource-recommendations')
            ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '15px',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Detailed Reasoning
            html.Div([
                html.H3("📝 Allocation Reasoning", style={'color': '#2C3E50', 'marginBottom': '15px'}),
                html.Div(id='allocation-reasoning')
            ], style={'backgroundColor': 'white', 'padding': '20px',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
    ]),
    
], style={'padding': '20px', 'backgroundColor': '#F5F6FA', 'fontFamily': 'Arial, sans-serif'})


# Main callback
@app.callback(
    [Output('parameter-status', 'children'),
     Output('game-state-table', 'data'),
     Output('alert-panel', 'children'),
     Output('heatmap-chart', 'figure'),
     Output('forecast-chart', 'figure'),
     Output('forecast-reasoning', 'children'),
     Output('resource-recommendations', 'children'),
     Output('allocation-reasoning', 'children'),
     Output('save-state-status', 'children')],
    [Input('update-btn', 'n_clicks'),
     Input('simulate-btn', 'n_clicks'),
     Input('save-state-btn', 'n_clicks')],
    [State('current-hour', 'value'),
     State('total-staff', 'value'),
     State('total-beds', 'value'),
     State('game-state-table', 'data')]
)
def update_dashboard(update_clicks, simulate_clicks, save_clicks, 
                    current_hour, total_staff, total_beds, state_data):
    """Main dashboard update with parameter controls"""
    
    # Determine trigger
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'update-btn'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Update game state
    game_state['current_hour'] = current_hour
    game_state['total_staff'] = total_staff
    game_state['total_beds'] = total_beds
    
    # Handle state saving
    save_status = ""
    if button_id == 'save-state-btn' and state_data:
        for row in state_data:
            dept_name = row['department']
            dept_key = next((k for k, v in DEPT_NAMES.items() if v == dept_name), None)
            if dept_key:
                game_state['current_patients'][dept_key] = int(row.get('current_patients', 0))
        save_status = "✓ Game state saved!"
    
    # Simulate next hour if requested
    if button_id == 'simulate-btn':
        # Generate new arrivals using Poisson distribution
        new_arrivals = generator.generate_real_time_data(current_hour, historical_data)
        
        # Process each department: discharge some patients, add new arrivals
        for dept in DEPT_NAMES.keys():
            # 1. Discharge/treat current patients
            current = game_state['current_patients'][dept]
            remaining_after_treatment = generator.simulate_patient_discharge(current, dept)
            
            # 2. Add new arrivals
            game_state['current_patients'][dept] = remaining_after_treatment + new_arrivals[dept]
            
            # Track total treated
            treated = current - remaining_after_treatment
            game_state['total_treated'] += treated
        
        # Advance hour
        current_hour = min(24, current_hour + 1)
        game_state['current_hour'] = current_hour
    
    # Parameter status
    param_status = html.Span([
        f"✓ Hour {current_hour}/24 | ",
        f"Staff: {total_staff} | ",
        f"Beds: {total_beds} | ",
        f"System configured"
    ])
    
    # Game state table data
    state_table_data = []
    for dept, dept_name in DEPT_NAMES.items():
        state_table_data.append({
            'department': dept_name,
            'current_patients': game_state['current_patients'][dept]
        })
    
    # Get forecasts for current hour
    forecasts = analytics.forecast_all_departments(current_hour)
    
    # Detect surges
    alerts = analytics.detect_surge(forecasts)
    
    # Alert panel with collapsible reasoning
    alert_components = []
    if alerts:
        for alert in alerts:
            color = '#E74C3C' if alert['severity'] == 'HIGH' else '#F39C12'
            alert_components.append(
                html.Div([
                    html.Div([
                        html.Span(f"⚠️  {alert['severity']}: {alert['department'].replace('_', ' ').title()}: ", 
                                 style={'fontWeight': 'bold', 'fontSize': '16px'}),
                        html.Span(f"Expected {alert['forecast']:.1f} patients ", 
                                 style={'fontSize': '16px'}),
                        html.Span(f"(threshold: {alert['threshold']:.1f})", 
                                 style={'fontSize': '14px', 'fontStyle': 'italic'})
                    ], style={'marginBottom': '10px'}),
                    html.Details([
                        html.Summary("📊 View detailed reasoning", 
                                   style={'cursor': 'pointer', 'fontSize': '14px', 
                                         'color': 'white', 'opacity': '0.9'}),
                        html.Div([
                            html.Pre(alert['reasoning'], 
                                    style={'fontSize': '13px', 'color': 'white', 'margin': '10px 0 0 0',
                                          'fontFamily': 'system-ui, -apple-system, sans-serif',
                                          'lineHeight': '1.6', 'whiteSpace': 'pre-wrap'})
                        ], style={'marginTop': '10px', 'paddingTop': '10px', 
                                 'borderTop': '1px solid rgba(255,255,255,0.3)'})
                    ])
                ], style={'backgroundColor': color, 'color': 'white', 'padding': '15px', 
                         'marginBottom': '10px', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.15)'})
            )
    else:
        alert_components.append(
            html.Div("✅ All departments within normal capacity", 
                    style={'backgroundColor': '#27AE60', 'color': 'white', 'padding': '15px',
                          'borderRadius': '8px', 'fontWeight': 'bold', 'fontSize': '16px'})
        )
    
    # Get resource allocation
    allocation_result = analytics.optimize_resource_allocation(
        forecasts, total_staff, total_beds
    )
    
    # Create heat map showing current utilization
    heatmap_data = []
    heatmap_depts = []
    for dept in DEPT_NAMES.keys():
        alloc = allocation_result['allocations'][dept]
        heatmap_data.append(alloc['utilization'] * 100)  # Convert to percentage
        heatmap_depts.append(DEPT_NAMES[dept])
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[heatmap_data],
        x=heatmap_depts,
        y=['Utilization %'],
        colorscale=[
            [0, '#27AE60'],    # Green (0%)
            [0.6, '#F39C12'],  # Yellow (60%)
            [0.85, '#E74C3C']  # Red (85%+)
        ],
        text=[[f"{val:.0f}%" for val in heatmap_data]],
        texttemplate='%{text}',
        textfont={"size": 14, "color": "white"},
        showscale=True,
        colorbar=dict(title="Utilization %")
    ))
    
    heatmap_fig.update_layout(
        height=150,
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title="",
        yaxis_title=""
    )
    
    # Forecast chart (next 4 hours)
    forecast_hours = list(range(current_hour, min(current_hour + 4, 25)))
    future_forecasts = analytics.forecast_next_n_rounds(current_hour, n=len(forecast_hours))
    
    forecast_fig = go.Figure()
    for dept, dept_name in DEPT_NAMES.items():
        forecast_values = [future_forecasts[h][dept]['forecast'] for h in forecast_hours]
        lower_bounds = [future_forecasts[h][dept]['lower_bound'] for h in forecast_hours]
        upper_bounds = [future_forecasts[h][dept]['upper_bound'] for h in forecast_hours]
        
        # Main line
        forecast_fig.add_trace(go.Scatter(
            x=forecast_hours,
            y=forecast_values,
            name=dept_name,
            mode='lines+markers',
            line=dict(color=DEPT_COLORS[dept], width=3),
            marker=dict(size=8)
        ))
        
        # Confidence band
        forecast_fig.add_trace(go.Scatter(
            x=forecast_hours + forecast_hours[::-1],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor=DEPT_COLORS[dept],
            line=dict(color='rgba(255,255,255,0)'),
            opacity=0.2,
            name=f'{dept_name} 95% CI',
            showlegend=False
        ))
    
    forecast_fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Predicted Patients (Poisson Model)",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Forecast reasoning with collapsible details
    forecast_reasoning_components = []
    for dept, forecast_data in forecasts.items():
        forecast_reasoning_components.append(
            html.Details([
                html.Summary([
                    html.Span(DEPT_NAMES[dept], 
                             style={'fontWeight': 'bold', 'fontSize': '16px', 'color': DEPT_COLORS[dept]}),
                    html.Span(f" - Forecast: {forecast_data['forecast']:.1f} patients", 
                             style={'fontSize': '15px', 'marginLeft': '10px', 'color': '#2C3E50'})
                ], style={'cursor': 'pointer', 'padding': '12px', 'backgroundColor': '#F8F9FA',
                         'borderRadius': '5px', 'marginBottom': '8px'}),
                html.Div([
                    html.Pre(forecast_data['reasoning'], 
                            style={'fontSize': '14px', 'color': '#34495E', 'marginTop': '10px',
                                  'whiteSpace': 'pre-wrap', 'backgroundColor': 'white', 
                                  'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #E0E0E0',
                                  'fontFamily': 'system-ui, -apple-system, sans-serif', 'lineHeight': '1.7'})
                ])
            ], style={'marginBottom': '12px'})
        )
    
    # Get resource allocation
    allocation_result = analytics.optimize_resource_allocation(
        forecasts, total_staff, total_beds
    )
    
    # Resource recommendations display
    resource_components = []
    for dept, alloc in allocation_result['allocations'].items():
        resource_components.append(
            html.Div([
                html.Div([
                    html.Span(DEPT_NAMES[dept], style={'fontWeight': 'bold', 'fontSize': '14px'}),
                    html.Span(f" (Forecast: {alloc['forecast']:.1f} patients)", 
                             style={'fontSize': '11px', 'color': '#7F8C8D', 'marginLeft': '10px'})
                ]),
                html.Div([
                    html.Span(f"👥 Staff: ", style={'fontSize': '13px'}),
                    html.Span(f"{alloc['staff_allocated']}", 
                             style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#3498DB'}),
                    html.Span(f"   🛏️ Beds: ", style={'fontSize': '13px', 'marginLeft': '20px'}),
                    html.Span(f"{alloc['beds_allocated']}", 
                             style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#2ECC71'}),
                ]),
                html.Div([
                    html.Span(f"Utilization: {alloc['utilization']:.1%} | ", 
                             style={'fontSize': '11px', 'color': '#7F8C8D'}),
                    html.Span(f"Est. Wait: {alloc['expected_wait_minutes']:.1f} min", 
                             style={'fontSize': '11px', 'color': '#E74C3C'})
                ])
            ], style={'padding': '12px', 'marginBottom': '10px', 'backgroundColor': '#F8F9FA',
                     'borderLeft': f'4px solid {DEPT_COLORS[dept]}', 'borderRadius': '5px'})
        )
    
    # Allocation reasoning with better styling
    allocation_reasoning_components = [
        html.Div([
            html.H4("🎯 System-Wide Strategy", style={'color': '#2C3E50', 'marginBottom': '10px', 'fontSize': '18px'}),
            html.Pre(allocation_result['system_reasoning'],
                    style={'fontSize': '14px', 'color': '#34495E', 'whiteSpace': 'pre-wrap',
                          'backgroundColor': '#E8F8F5', 'padding': '18px', 'borderRadius': '8px',
                          'border': '2px solid #27AE60', 'marginBottom': '20px',
                          'fontFamily': 'system-ui, -apple-system, sans-serif', 'lineHeight': '1.7'})
        ])
    ]
    
    # Individual department reasoning - collapsible
    for dept, alloc in allocation_result['allocations'].items():
        allocation_reasoning_components.append(
            html.Details([
                html.Summary(f"📋 {DEPT_NAMES[dept]} - Detailed Reasoning",
                           style={'fontWeight': 'bold', 'cursor': 'pointer', 'fontSize': '15px',
                                 'color': DEPT_COLORS[dept], 'padding': '10px', 
                                 'backgroundColor': '#F8F9FA', 'borderRadius': '5px'}),
                html.Pre(alloc['reasoning'],
                        style={'fontSize': '13px', 'color': '#34495E', 'whiteSpace': 'pre-wrap',
                              'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px',
                              'marginTop': '10px', 'border': '1px solid #E0E0E0',
                              'fontFamily': 'system-ui, -apple-system, sans-serif', 'lineHeight': '1.7'})
            ], style={'marginBottom': '12px'})
        )
    
    return (param_status, state_table_data, html.Div(alert_components), 
            heatmap_fig, forecast_fig, html.Div(forecast_reasoning_components),
            html.Div(resource_components), html.Div(allocation_reasoning_components),
            save_status)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 ENHANCED ER COMMAND CENTER")
    print("="*70)
    print("\nNew Features:")
    print("  ✓ Adjustable parameters (staff, beds, round)")
    print("  ✓ Edit game state (set current patients)")
    print("  ✓ Poisson distribution forecasting")
    print("  ✓ Reasoning-backed recommendations")
    print("  ✓ Combined staff + bed allocation")
    print("  ✓ M/M/c queue theory integration")
    print("\nDashboard: http://127.0.0.1:8050")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
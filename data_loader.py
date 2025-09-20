import pandas as pd
from datetime import datetime
import numpy as np

def load_and_preprocess_data(filepath):
    """Load and preprocess sports data"""
    try:
        # For football data format
        df = pd.read_csv(filepath)
        
        # Basic preprocessing
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date')
        
        # Create features
        df['HomeWin'] = (df['FTHG'] > df['FTAG']).astype(int)
        df['Draw'] = (df['FTHG'] == df['FTAG']).astype(int)
        df['AwayWin'] = (df['FTHG'] < df['FTAG']).astype(int)
        
        # Team performance metrics
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        team_stats = {}
        
        for team in teams:
            home_games = df[df['HomeTeam'] == team]
            away_games = df[df['AwayTeam'] == team]
            
            team_stats[team] = {
                'home_win_rate': home_games['HomeWin'].mean() if len(home_games) > 0 else 0.5,
                'away_win_rate': away_games['AwayWin'].mean() if len(away_games) > 0 else 0.5,
                'avg_goals_scored': np.mean([
                    home_games['FTHG'].mean() if len(home_games) > 0 else 0,
                    away_games['FTAG'].mean() if len(away_games) > 0 else 0
                ])
            }
        
        return df, team_stats
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create sample data if loading fails
        return create_sample_data(), {}

def create_sample_data():
    """Create sample data if real data unavailable"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
    
    data = []
    for i in range(100):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_task])
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        
        data.append({
            'Date': dates[i],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': home_goals,
            'FTAG': away_goals
        })
    
    return pd.DataFrame(data)

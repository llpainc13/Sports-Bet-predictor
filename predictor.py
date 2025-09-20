import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class SportsPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def prepare_features(self, df, team_stats):
        """Prepare features for training"""
        features = []
        targets = []
        
        for _, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Get team stats
            home_stats = team_stats.get(home_team, {'home_win_rate': 0.5, 'avg_goals_scored': 1.5})
            away_stats = team_stats.get(away_team, {'away_win_rate': 0.3, 'avg_goals_scored': 1.2})
            
            # Create feature vector
            feature_vector = [
                home_stats['home_win_rate'],
                away_stats['away_win_rate'],
                home_stats['avg_goals_scored'],
                away_stats['avg_goals_scored'],
                1 if 'Home' in str(row.get('Venue', 'Home')) else 0  # Home advantage
            ]
            
            features.append(feature_vector)
            targets.append(row['HomeWin'] * 0 + row['Draw'] * 1 + row['AwayWin'] * 2)
        
        return np.array(features), np.array(targets)
    
    def train(self, df, team_stats):
        """Train the prediction model"""
        X, y = self.prepare_features(df, team_stats)
        
        if len(X) > 10:  # Need minimum data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            return accuracy
        else:
            self.is_trained = False
            return 0.0
    
    def predict(self, home_team_stats, away_team_stats, is_home_advantage=True):
        """Make prediction for a match"""
        if not self.is_trained:
            # Return random prediction if not trained
            return {
                'home_win': 0.33,
                'draw': 0.33,
                'away_win': 0.34,
                'confidence': 0.5
            }
        
        # Prepare feature vector
        features = np.array([[
            home_team_stats['home_win_rate'],
            away_team_stats['away_win_rate'],
            home_team_stats['avg_goals_scored'],
            away_team_stats['avg_goals_scored'],
            1 if is_home_advantage else 0
        ]])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            'home_win': float(probabilities[0]) if len(probabilities) > 0 else 0.33,
            'draw': float(probabilities[1]) if len(probabilities) > 1 else 0.33,
            'away_win': float(probabilities[2]) if len(probabilities) > 2 else 0.34,
            'confidence': float(np.max(probabilities))
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

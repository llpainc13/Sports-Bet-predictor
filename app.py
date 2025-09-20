from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import os
from models.data_loader import load_and_preprocess_data
from models.predictor import SportsPredictor
import json

app = Flask(__name__, static_folder='../frontend')

# Global variables
df = None
team_stats = None
predictor = SportsPredictor()

# Load data on startup
@app.before_first_request
def load_data():
    global df, team_stats, predictor
    try:
        data_path = 'data/sample_data.csv'
        if os.path.exists(data_path):
            df, team_stats = load_and_preprocess_data(data_path)
            accuracy = predictor.train(df, team_stats)
            print(f"Model trained with accuracy: {accuracy:.2f}")
        else:
            print("Sample data not found, using dummy data")
    except Exception as e:
        print(f"Error loading data: {e}")

# Serve frontend files
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('../frontend', 'dashboard.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('../frontend', filename)

# API endpoints
@app.route('/api/teams')
def get_teams():
    if team_stats:
        return jsonify(list(team_stats.keys()))
    return jsonify([])

@app.route('/api/team-stats/<team_name>')
def get_team_stats(team_name):
    if team_stats and team_name in team_stats:
        return jsonify(team_stats[team_name])
    return jsonify({'error': 'Team not found'}), 404

@app.route('/api/predict', methods=['POST'])
def predict_match():
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Home and away teams required'}), 400
        
        # Get team stats
        home_stats = team_stats.get(home_team, {
            'home_win_rate': 0.5, 
            'away_win_rate': 0.3, 
            'avg_goals_scored': 1.5
        })
        
        away_stats = team_stats.get(away_team, {
            'home_win_rate': 0.4, 
            'away_win_rate': 0.3, 
            'avg_goals_scored': 1.2
        })
        
        # Make prediction
        prediction = predictor.predict(home_stats, away_stats)
        
        return jsonify({
            'home_team': home_team,
            'away_team': away_team,
            'prediction': prediction
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/matches')
def get_matches():
    """Get recent matches for display"""
    try:
        if df is not None:
            recent_matches = df.tail(10)[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].to_dict('records')
            return jsonify(recent_matches)
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

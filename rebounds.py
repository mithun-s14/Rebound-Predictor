import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, teamgamelog, LeagueDashTeamStats, commonplayerinfo
from nba_api.stats.static import players, teams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Collect Player Game Logs
def get_player_game_logs(player_name, season):
    # Find player ID
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        raise ValueError(f"Player '{player_name}' not found.")
    player_id = player_dict[0]['id']

    # Fetch player game logs
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    gamelog_df = gamelog.get_data_frames()[0]
    return gamelog_df

# Collect Team Game Logs
def get_team_game_logs(team_name, season):
    # Find team ID
    team_dict = teams.find_teams_by_full_name(team_name)
    if not team_dict:
        raise ValueError(f"Team '{team_name}' not found.")
    team_id = team_dict[0]['id']

    # Fetch team game logs
    team_gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
    team_gamelog_df = team_gamelog.get_data_frames()[0]
    return team_gamelog_df

# Combine Data into a Single DataFrame
def prepare_dataset(player_name, team_name, season):
    # Fetch player game logs
    player_logs = get_player_game_logs(player_name, season)
    player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])

    # Fetch team game logs
    team_logs = get_team_game_logs(team_name, season)
    team_logs['GAME_DATE'] = pd.to_datetime(team_logs['GAME_DATE'])

    # Merge player logs with team logs
    df = pd.merge(
        player_logs,
        team_logs[['GAME_DATE', 'REB']],  # Use team rebounds as opponent rebounds allowed
        left_on='GAME_DATE',
        right_on='GAME_DATE',
        how='left',
        suffixes=('', '_OPP')
    )

    # Feature Engineering
    df['HOME_AWAY'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['REBOUNDS'] = df['REB']  # Target variable

    # Select relevant features
    features = [
        'PTS', 'AST', 'REB', 'MIN', 'HOME_AWAY', 'REB_OPP'
    ]
    df = df[features + ['REBOUNDS']].dropna()

    return df

def train_model(df):
    # Define features and target
    X = df.drop(columns=['REBOUNDS'])
    y = df['REBOUNDS']

    # Preprocessing pipeline
    numerical_features = ['PTS', 'AST', 'REB', 'MIN', 'REB_OPP']
    categorical_features = ['HOME_AWAY']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")

    # Add predictions to the test set for analysis
    # test_results = X_test.copy()
    # test_results['ACTUAL_REBOUNDS'] = y_test
    # test_results['PREDICTED_REBOUNDS'] = y_pred
    # print("\nTest Set Results:")
    # print(test_results.head())

    return model

def calculate_player_averages(player_logs, num_games):
    # Calculate rolling averages for the last num_games
    player_logs = player_logs.sort_values('GAME_DATE', ascending=False)
    averages = player_logs[['PTS', 'AST', 'REB', 'MIN']].head(num_games).mean().to_dict()
    return averages

def predict_next_matchup(model, player_averages, opponent_rebounds_allowed, home_away):
    # Create a DataFrame for the next matchup
    next_matchup = pd.DataFrame({
        'PTS': [player_averages['PTS']],
        'AST': [player_averages['AST']],
        'REB': [player_averages['REB']],
        'MIN': [player_averages['MIN']],
        'HOME_AWAY': [home_away],
        'REB_OPP': [opponent_rebounds_allowed]
    })

    # Make prediction
    predicted_rebounds = model.predict(next_matchup)
    return predicted_rebounds[0]


def get_rebounds_allowed(team_name):
    # Fetch team defensive stats
    team_stats = LeagueDashTeamStats(per_mode_detailed='PerGame', measure_type_detailed_defense='Opponent')

    # Convert to DataFrame
    df = team_stats.get_data_frames()[0]

    # Select relevant columns
    df = df[['TEAM_NAME', 'OPP_REB']]

    team_row = df[df['TEAM_NAME'].str.lower() == team_name.lower()]
    if not team_row.empty:
        return team_row.iloc[0]['OPP_REB']
    else:
        return f"Team '{team_name}' not found."
    
def get_player_team(player_name):
    # Search for player
    player = players.find_players_by_full_name(player_name)

    if not player:
        return f"Player '{player_name}' not found."

    player_id = player[0]['id']  # Get first matching player's ID

    # Fetch player details
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    df = player_info.get_data_frames()[0]

    # Extract team name
    team_name = df.loc[0, 'TEAM_NAME']

    return team_name if team_name else "No team (Free Agent/Injured/Retired)"

def main():
    season = "2024-25"

    # Input for next matchup
    print("\nEnter details for the next matchup:")
    player_name = input("Enter player name: ")
    team_name = get_player_team(player_name)
    opponent = input("Enter opponent team name: ")
    opponent_rebounds_allowed = get_rebounds_allowed(opponent)
    home_away = int(input("Is the game at home? (1 for Home, 0 for Away): "))

    # Prepare dataset
    df = prepare_dataset(player_name, team_name, season)
    # print("Dataset Prepared:")
    # print(df.head())

    # Train the model and get test results
    model = train_model(df)

    # Calculate player averages for the last 5 games
    player_logs = get_player_game_logs(player_name, season)
    player_averages = calculate_player_averages(player_logs, num_games=20)

    # Predict rebounds for the next matchup
    predicted_rebounds = predict_next_matchup(model, player_averages, opponent_rebounds_allowed, home_away)
    print(f"\nPredicted Rebounds for Next Matchup: {predicted_rebounds}")

if __name__ == "__main__":
    main()
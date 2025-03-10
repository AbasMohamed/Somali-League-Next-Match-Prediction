import streamlit as st
import pandas as pd
import pickle
import os
from scipy.stats import poisson
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler


# Load df_team_strength from pickle
with open('df_team_strength.pkl', 'rb') as file:
    df_team_strength = pickle.load(file)

# Ensure index is clean
df_team_strength.index = df_team_strength.index.str.replace(r'\s+', ' ', regex=True).str.strip()

# Load dataset
import os
import pandas as pd

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Cleaned4_Somali_league_datasets.csv")

# Load CSV
df = pd.read_csv(file_path)
print("✅ CSV Loaded Successfully!")





# Load models and scaler
with open('logistic_regression.pkl', 'rb') as f:
    log_reg = pickle.load(f)
with open('random_forest.pkl', 'rb') as f:
    random_forest = pickle.load(f)
with open('xgboost.pkl', 'rb') as f:
    xgb = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define feature columns
feature_columns = [
    'Home_Score', 'Away_Score', 'Home_Goal_Diff', 'Away_Goal_Diff', 
    'Home_Last_5_Wins', 'Away_Last_5_Wins', 'Home_Last_5_Points', 
    'Away_Last_5_Points', 'H2H_Home_Wins', 'H2H_Home_Losses', 
    'H2H_Away_Wins', 'H2H_Away_Losses', 'H2H_Draws'
]


# Functions for validation and data processing
def validate_teams(home_team, away_team, df):
    home_exists = home_team in df['Home_Team'].values or home_team in df['Away_Team'].values
    away_exists = away_team in df['Away_Team'].values or away_team in df['Home_Team'].values
    if not home_exists:
        print(f"Error: '{home_team}' is not a valid team in the Somali League dataset.")
        return False
    if not away_exists:
        print(f"Error: '{away_team}' is not a valid team in the Somali League dataset.")
        return False
    if home_team == away_team:
        print("Error: Home Team and Away Team cannot be the same.")
        return False
    return True

def get_h2h_info(home_team, away_team, df):
    h2h_matches = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                     ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))]
    
    if h2h_matches.empty:
        return {'total_matches': 0, 'home_wins': 0, 'away_wins': 0, 'draws': 0}
    
    h2h_home = df[(df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)]
    if not h2h_home.empty:
        latest_h2h = h2h_home.iloc[-1]
        home_wins = int(latest_h2h['H2H_Home_Wins'])
        home_losses = int(latest_h2h['H2H_Home_Losses'])
        away_wins = int(latest_h2h['H2H_Away_Wins'])
        away_losses = int(latest_h2h['H2H_Away_Losses'])
        draws = int(latest_h2h['H2H_Draws'])
        total_matches = home_wins + home_losses + draws
    else:
        total_matches = len(h2h_matches)
        home_wins_as_home = len(h2h_matches[(h2h_matches['Home_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 0)])
        away_wins_as_away = len(h2h_matches[(h2h_matches['Away_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 2)])
        draws = len(h2h_matches[h2h_matches['Result_Encoded'] == 1])
        home_wins = home_wins_as_home + len(h2h_matches[(h2h_matches['Home_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 2)])
        away_wins = away_wins_as_away + len(h2h_matches[(h2h_matches['Home_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 2)])

    return {'total_matches': total_matches, 'home_wins': home_wins, 'away_wins': away_wins, 'draws': draws}

def get_team_form(team, df):
    team_matches = df[(df['Home_Team'] == team) | (df['Away_Team'] == team)].sort_index(ascending=False).head(5)
    
    if team_matches.empty:
        return {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0}
    
    wins = 0
    draws = 0
    losses = 0
    points = 0
    
    for _, match in team_matches.iterrows():
        if match['Home_Team'] == team:
            if match['Result_Encoded'] == 0:  # Home win
                wins += 1
                points += 3
            elif match['Result_Encoded'] == 1:  # Draw
                draws += 1
                points += 1
            else:  # Away win (loss for home team)
                losses += 1
        else:  # Team is Away_Team
            if match['Result_Encoded'] == 2:  # Away win
                wins += 1
                points += 3
            elif match['Result_Encoded'] == 1:  # Draw
                draws += 1
                points += 1
            else:  # Home win (loss for away team)
                losses += 1
    
    return {'wins': wins, 'draws': draws, 'losses': losses, 'points': points}

def get_recent_match_result(home_team, away_team, df):
    h2h_matches = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                     ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))].sort_index(ascending=False)
    
    if h2h_matches.empty:
        return None
    
    latest_match = h2h_matches.iloc[0]
    home_team_match = latest_match['Home_Team']
    away_team_match = latest_match['Away_Team']
    home_score = int(latest_match['Home_Score'])
    away_score = int(latest_match['Away_Score'])
    season = latest_match['Season']
    
    return {'home_team': home_team_match, 'away_team': away_team_match, 'home_score': home_score, 'away_score': away_score, 'season': season}

def prepare_input(home_team, away_team, df, feature_columns):
    home_data = df[df['Home_Team'] == home_team]
    away_data = df[df['Away_Team'] == away_team]
    
    if home_data.empty:
        home_data = df
        print(f"Warning: No historical data for {home_team} as Home Team. Using dataset averages.")
    if away_data.empty:
        away_data = df
        print(f"Warning: No historical data for {away_team} as Away Team. Using dataset averages.")
    
    home_means = home_data.mean(numeric_only=True)
    away_means = away_data.mean(numeric_only=True)
    
    features_dict = {}
    for col in feature_columns:
        if 'Home' in col:
            features_dict[col] = home_means.get(col, df[col].mean()) if col in home_means else df[col].mean()
        elif 'Away' in col:
            features_dict[col] = away_means.get(col, df[col].mean()) if col in away_means else df[col].mean()
        else:
            features_dict[col] = (home_means.get(col, 0) + away_means.get(col, 0)) / 2 if col in home_means else df[col].mean()
    
    input_data = pd.DataFrame([features_dict], columns=feature_columns)
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

def interpret_prediction(pred):
    if pred == 0:
        return "Home Team Wins"
    elif pred == 1:
        return "Draw"
    else:
        return "Away Team Wins"


def predict_points(home, away):
    if home in df_team_strength.index and away in df_team_strength.index:
        lamb_home = df_team_strength.at[home, 'GoalsScored'] * df_team_strength.at[away, 'GoalsConceded']
        lamb_away = df_team_strength.at[away, 'GoalsScored'] * df_team_strength.at[home, 'GoalsConceded']
        prob_home, prob_away, prob_draw = 0, 0, 0
        
        for x in range(0, 11):  # Number of goals home team
            for y in range(0, 11):  # Number of goals away team
                p = poisson.pmf(x, lamb_home) * poisson.pmf(y, lamb_away)
                if x == y:
                    prob_draw += p
                elif x > y:
                    prob_home += p
                else:
                    prob_away += p
        
        points_home = 3 * prob_home + prob_draw
        points_away = 3 * prob_away + prob_draw
        return points_home, points_away
    else:
        return 0, 0



# Function to get team logo path
def get_team_logo(team_name):
    logo_folder = "TeamLogo"
    
    # Rename specific teams
    if team_name == "Heegan S.C":
        team_name = "Heegan SC"
    elif team_name == "Horseed S.C":
        team_name = "Horseed SC"
    
    # Check for available formats
    for ext in ["png", "jpeg", "jpg"]:
        logo_path = os.path.join(logo_folder, f"{team_name}.{ext}")
        if os.path.exists(logo_path):
            return logo_path
    
    return None  # Return None if no logo is found

# Function to create circular logos
def create_circular_logo(image_path, size=(150, 150)):
    image = Image.open(image_path).convert("RGBA")
    image = image.resize(size, Image.LANCZOS)  # Resize to fixed size
    
    # Create circular mask
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    
    # Apply circular mask
    circular_image = Image.new("RGBA", size, (255, 255, 255, 0))  # Transparent background
    circular_image.paste(image, (0, 0), mask)
    
    return circular_image

# Streamlit UI
st.title("Somali Football Match Predictor")

team_names = ["Select a team"] + df_team_strength.index.tolist()


# Initialize session state to handle auto-selection of the away team
if "home_team" not in st.session_state:
    st.session_state.home_team = "Select a team"

if "away_team" not in st.session_state:
    st.session_state.away_team = "Select a team"

# Display team logos above the selectboxes
col1, col2 = st.columns(2)

with col1:
    home_logo = get_team_logo(st.session_state.home_team) if st.session_state.home_team != "Select a team" else None
    if home_logo:
        st.image(create_circular_logo(home_logo), caption=st.session_state.home_team, use_container_width=False)

with col2:
    away_logo = get_team_logo(st.session_state.away_team) if st.session_state.away_team != "Select a team" else None
    if away_logo:
        st.image(create_circular_logo(away_logo), caption=st.session_state.away_team, use_container_width=False)

# Create selectboxes below the team logos
sel1, sel2 = st.columns(2)

with sel1:
    home_team = st.selectbox("Select Home Team", options=team_names, index=team_names.index(st.session_state.home_team), key='home_team')

# Automatically update away team based on selected home team
if home_team != "Select a team":
    away_team_options = ["Select a team"] + [team for team in df_team_strength.index if team != home_team]
    
    # If the current away team is the same as the home team, reset it
    if st.session_state.away_team == home_team or st.session_state.away_team not in away_team_options:
        st.session_state.away_team = away_team_options[1] if len(away_team_options) > 1 else "Select a team"

else:
    away_team_options = ["Select a team"] + df_team_strength.index.tolist()

with sel2:
    away_team = st.selectbox("Select Away Team", options=away_team_options, index=away_team_options.index(st.session_state.away_team), key='away_team', disabled=False)
# Initialize session state if not already set
if "home_team" not in st.session_state:
    st.session_state.home_team = "Select a team"

if "away_team" not in st.session_state:
    st.session_state.away_team = "Select a team"

# Function to clear selections
def clear_selection():
    st.session_state.home_team = "Select a team"
    st.session_state.away_team = "Select a team"
btn1, btn2 = st.columns(2)
with btn1:
    # Button to predict
    if st.button("Predict Match Outcome"):
     
        if home_team != "Select a team" and away_team != "Select a team" and validate_teams(home_team, away_team, df):
            h2h_info = get_h2h_info(home_team, away_team, df)
            home_form = get_team_form(home_team, df)
            away_form = get_team_form(away_team, df)

            input_data = prepare_input(home_team, away_team, df, feature_columns)

            log_reg_pred = log_reg.predict(input_data)[0]
            rf_pred = random_forest.predict(input_data)[0]
            xgb_pred = xgb.predict(input_data)[0] - 1

            if h2h_info['total_matches'] == 0 :
                # Assuming 'points_home' and 'points_away' are the predicted points for the teams
                # and 'home_team' and 'away_team' are the team names

                # Convert the predicted points into a percentage for each team
                points_home, points_away = predict_points(home_team, away_team)
                

                # Assuming 'points_home' and 'points_away' are the predicted points for the teams
                # and 'home_team' and 'away_team' are the team names

                # Convert the predicted points into a percentage for each team
                home_team_win_percent = max(0, min(100, points_home * 33.33))  # Scale the points to percentage (0-100%)
                away_team_win_percent = max(0, min(100, points_away * 33.33))  # Scale the points to percentage (0-100%)

                # Determine the outcome as win, draw, or loss based on the points range
                def outcome_from_points(points):
                    if points >= 2:
                        return "Win"
                    elif points == 1:
                        return "Draw"
                    else:
                        return "Loss"

                home_team_outcome = outcome_from_points(points_home)
                away_team_outcome = outcome_from_points(points_away)

               # Display the message about no head-to-head record in one line
                st.markdown(
                    f"""
                    <div style="text-align: center; font-size: 16px; white-space: nowrap;">
                        ⚠️ It looks like <b>{home_team}</b> and <b>{away_team}</b> have never faced each other in a head-to-head <br> match before. 
                        However, based on team strength and performance data, we can predict the expected outcome.<br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Force both results to be on one line and shift them slightly to the right
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: nowrap; margin-right: -340px;">
                        <div style="background-color: #dff0d8; padding: 12px; border-radius: 8px; text-align: center; white-space: nowrap;">
                            🏆 <b>{home_team}: {home_team_win_percent:.2f}% chance of {home_team_outcome}</b>
                        </div>
                        <div style="background-color: #d9edf7; padding: 12px; border-radius: 8px; text-align: center; white-space: nowrap;">
                            ⚽ <b>{away_team}: {away_team_win_percent:.2f}% chance of {away_team_outcome}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )




                


            else:
                st.subheader("Team Form & Head-to-Head Stats")
                st.write(f"⚽  **{home_team} Last 5 Matches:** {home_form['wins']} Wins, {home_form['draws']} Draws, {home_form['losses']} Losses, {home_form['points']} points")
                st.write(f"⚽  **{away_team} Last 5 Matches:** {away_form['wins']} Wins, {away_form['draws']} Draws, {away_form['losses']} Losses, {away_form['points']} points")
                st.write(f"⚔️ **Head-to-Head:** Total: {h2h_info['total_matches']} | {home_team} Wins: {h2h_info['home_wins']} | {away_team} Wins: {h2h_info['away_wins']} | Draws: {h2h_info['draws']}")
                recent_match = get_recent_match_result(home_team, away_team, df)
                if recent_match:
                    st.write(f"⚔️ **Most Recent Match Result:** season: {recent_match['season']} | {home_team}  {recent_match['home_score']} - {recent_match['away_score']}   {away_team} ")
                else:
                    print("No previous matches found between these teams.")
            
                st.subheader("Match Prediction Results")
        
                st.write("🏆 **Logistic Regression** : ", interpret_prediction(log_reg_pred))
                st.write("🏆 **Random Forest** : ", interpret_prediction(rf_pred))
                st.write("🏆 **XGBoost** : ", interpret_prediction(xgb_pred))

            
        else:
            st.error("Please select valid teams.")
    
with btn2:
    st.button("Clear Selection", on_click=clear_selection)






import streamlit as st
import pandas as pd
import re
import datetime
from io import StringIO
import os 
from dotenv import load_dotenv


from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAI



@st.cache_data
def load_matches():
    df = pd.read_csv("data/df_matches.csv", parse_dates=["date"])
    df['season'] = df['season'].astype(str).str.strip()
    for col in ['team1', 'team2', 'winner']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

@st.cache_data
def load_deliveries():
    df = pd.read_csv("data\deliver_df.csv.gz")
    for col in ['batting_team', 'bowling_team', 'player_dismissed', 'batter', 'bowler']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def get_unique_teams(df_matches, df_deliveries):
    teams_matches = pd.concat([df_matches['team1'], df_matches['team2']]).unique()
    teams_deliveries = pd.concat([df_deliveries['batting_team'], df_deliveries['bowling_team']]).unique()
    teams = set([t for t in teams_matches if pd.notna(t)] + [t for t in teams_deliveries if pd.notna(t)])
    return list(teams)

def get_unique_match_types(df):
    if 'match_type' in df.columns:
        mts = df['match_type'].dropna().unique()
        return [m.lower() for m in mts]
    return []

def get_unique_players(df_deliveries):
    players = pd.concat([df_deliveries['batter'], df_deliveries['bowler'], df_deliveries['player_dismissed']])
    players = players.dropna().unique()
    players = [p.strip() for p in players if p.strip() != '']
    return players

def extract_filters(question, teams, match_types, players):
    q = question.lower()
    filters = {}

    for team in teams:
        if team.lower() in q:
            filters['team'] = team
            break

    for mt in match_types:
        if mt in q:
            filters['match_type'] = mt
            break

    seasons = [str(y) for y in range(2008, 2025)]
    for year in seasons:
        if year in q:
            filters['season'] = year
            break

    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    match = re.search(date_pattern, question)
    if match:
        filters['date'] = match.group(1)

    for player in players:
        pattern = r'\b' + re.escape(player.lower()) + r'\b'
        if re.search(pattern, q):
            filters['player'] = player
            break

    return filters

def filter_matches(df, filters):
    filtered = df.copy()

    if 'team' in filters:
        team = filters['team'].lower()
        filtered = filtered[
            (filtered['team1'].str.lower().str.contains(team, na=False)) |
            (filtered['team2'].str.lower().str.contains(team, na=False))
        ]

    if 'match_type' in filters:
        mt = filters['match_type'].lower()
        filtered = filtered[
            filtered['match_type'].str.lower().str.contains(mt, na=False)
        ]

    if 'season' in filters:
        filtered = filtered[
            filtered['season'].str.strip() == filters['season']
        ]

    if 'date' in filters:
        filtered['date_only'] = pd.to_datetime(filtered['date']).dt.date
        try:
            dt_obj = datetime.datetime.strptime(filters['date'], '%Y-%m-%d').date()
            filtered = filtered[filtered['date_only'] == dt_obj]
        except:
            pass

    return filtered

def filter_deliveries(df, filters):
    filtered = df.copy()

    if 'player' in filters:
        player = filters['player'].lower()
        filtered = filtered[
            (filtered['batter'].str.lower() == player) |
            (filtered['bowler'].str.lower() == player) |
            (filtered['player_dismissed'].str.lower() == player)
        ]

    if 'team' in filters:
        team = filters['team'].lower()
        filtered = filtered[
            (filtered['batting_team'].str.lower().str.contains(team, na=False)) |
            (filtered['bowling_team'].str.lower().str.contains(team, na=False))
        ]

    return filtered

# -- MAIN APP --

def main():
    st.title("IPL Q&A with Google Generative AI")

    df_matches = load_matches()
    df_deliveries = load_deliveries()

    teams = get_unique_teams(df_matches, df_deliveries)
    match_types = get_unique_match_types(df_matches)
    players = get_unique_players(df_deliveries)

    user_question = st.text_input("Ask your IPL question:")

    if user_question:
        filters = extract_filters(user_question, teams, match_types, players)
        st.markdown("### Extracted Filters:")
        st.json(filters)

        filtered_matches = filter_matches(df_matches, filters)

        if 'season' in filters:
            match_ids = filtered_matches['id'].tolist() if 'id' in filtered_matches.columns else []
            filtered_deliveries = df_deliveries[df_deliveries['match_id'].isin(match_ids)]
        else:
            filtered_deliveries = df_deliveries.copy()

        filtered_deliveries = filter_deliveries(filtered_deliveries, filters)

        st.markdown(f"### Filtered Matches: {len(filtered_matches)} rows")
        st.dataframe(filtered_matches)

        st.markdown(f"### Filtered Deliveries: {len(filtered_deliveries)} rows")
        st.dataframe(filtered_deliveries)

        if filtered_matches.empty and filtered_deliveries.empty:
            st.warning("No data found for your query filters.")
            return

        python_repl = PythonREPL()

        def execute_user_code(user_code):
            
            user_code_cleaned = re.sub(r"^```(?:python)?|```$", "", user_code.strip(), flags=re.MULTILINE).strip()

            
            csv_matches = filtered_matches.to_csv(index=False)
            csv_deliveries = filtered_deliveries.to_csv(index=False)

            
            wrapped_code = f'''
import pandas as pd
from io import StringIO

csv_matches = """{csv_matches}"""
csv_deliveries = """{csv_deliveries}"""

df_matches = pd.read_csv(StringIO(csv_matches))
df_deliveries = pd.read_csv(StringIO(csv_deliveries))

{user_code_cleaned}
'''

            # Run wrapped code inside PythonREPL
            return python_repl.run(wrapped_code)


        repl_tool = Tool(
            name="python_repl",
            description=(
                "Python REPL to execute code on filtered IPL dataframes df_matches and df_deliveries. "
                "Use print() to output your results."
            ),
            func=execute_user_code
        )

        api_key = st.secrets["api_keys"]["google_api_key"]


        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            max_output_tokens=1000,
            google_api_key=api_key 
        )

        agent = create_pandas_dataframe_agent(
            llm,
            [filtered_matches, filtered_deliveries],
            extra_tools=[repl_tool],
            handle_parsing_errors=True,
            verbose=True,
            allow_dangerous_code=True
        )

        st.markdown("### Answer from GoogleGenerativeAI:")
        with st.spinner("Generating answer..."):
            try:
                answer = agent.run(user_question)
                st.markdown(f"**{answer}**")
            except Exception as e:
                st.error(f"Error running agent: {e}")

if __name__ == "__main__":
    main()

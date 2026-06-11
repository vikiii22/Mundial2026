import pandas as pd
import numpy as np
import gzip
import csv
import os
from urllib.request import urlopen
from io import BytesIO

TEAM_NAME_MAP = {
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Turkey': 'Turkiye',
    'China PR': 'China',
    'DR Congo': 'Congo DR',
    'Hongkong': 'Hong Kong',
}

def load_squad_data():
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "national_teams_snapshot.csv")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        data_url = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/national_teams.csv.gz"
        try:
            req = urlopen(data_url)
            buf = BytesIO(req.read())
            with gzip.GzipFile(fileobj=buf) as f:
                df = pd.read_csv(f)
            df.to_csv(local_path, index=False)
            print(f"  -> Datos de plantilla guardados en {local_path}")
        except Exception as e:
            print(f"  -> No se pudo descargar datos de plantilla: {e}")
            return None

    squad = {}
    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        try:
            val = float(row.get("total_market_value", 0) or 0)
        except:
            val = 0.0
        try:
            fifa = int(row.get("fifa_ranking", 0) or 0)
        except:
            fifa = 0
        try:
            age = float(row.get("average_age", 0) or 0)
        except:
            age = 0.0
        try:
            size = int(row.get("squad_size", 0) or 0)
        except:
            size = 0
        squad[name] = {
            "fifa_ranking": fifa,
            "squad_value": val,
            "avg_age": age,
            "squad_size": size
        }

    return squad

def get_team_squad_data(team_name, squad):
    mapped = TEAM_NAME_MAP.get(team_name, team_name)
    if mapped in squad:
        return squad[mapped]
    return None

def add_squad_features(df, squad):
    features_team = ["fifa_ranking", "squad_value", "avg_age", "squad_size"]
    features_rival = ["rival_fifa_ranking", "rival_squad_value", "rival_avg_age", "rival_squad_size"]

    for ft in features_team:
        if ft in df.columns:
            df.drop(columns=[ft], inplace=True, errors="ignore")
    for fr in features_rival:
        if fr in df.columns:
            df.drop(columns=[fr], inplace=True, errors="ignore")

    team_data = df["team"].apply(lambda t: get_team_squad_data(t, squad))
    rival_data = df["opponent"].apply(lambda t: get_team_squad_data(t, squad))

    for ft in features_team:
        df[ft] = team_data.apply(lambda d: d[ft] if d else np.nan)
    for i, fr in enumerate(features_rival):
        df[fr] = rival_data.apply(lambda d: list(d.values())[i] if d else np.nan)

    df["log_squad_value"] = df["squad_value"].apply(lambda v: np.log10(v + 1) if pd.notna(v) else np.nan)
    df["rival_log_squad_value"] = df["rival_squad_value"].apply(lambda v: np.log10(v + 1) if pd.notna(v) else np.nan)

    return df

def get_extra_features():
    return [
        "fifa_ranking", "log_squad_value", "avg_age",
        "rival_fifa_ranking", "rival_log_squad_value", "rival_avg_age"
    ]

def get_foto_fija_squad_columns():
    return ["fifa_ranking", "squad_value", "avg_age", "squad_size"]

def get_foto_fija_rival_columns():
    return ["rival_fifa_ranking", "rival_squad_value", "rival_avg_age", "rival_squad_size"]

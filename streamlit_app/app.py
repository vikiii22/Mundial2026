import os, sys, json, time
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from src.clases_simulacion import Match, Tournament

MODEL_PATH = os.path.join(PROJECT_ROOT, "data/ai_models/xg_model_misterclaude.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/ai_models/xg_preds_J1_misterclaude_complete.csv")

FLAGS = {
    "Algeria": "\U0001f1e9\U0001f1ff", "Argentina": "\U0001f1e6\U0001f1f7", "Australia": "\U0001f1e6\U0001f1fa",
    "Austria": "\U0001f1e6\U0001f1f9", "Belgium": "\U0001f1e7\U0001f1ea", "Bosnia and Herzegovina": "\U0001f1e7\U0001f1e6",
    "Brazil": "\U0001f1e7\U0001f1f7", "Canada": "\U0001f1e8\U0001f1e6", "Cape Verde": "\U0001f1e8\U0001f1fb",
    "Colombia": "\U0001f1e8\U0001f1f4", "Croatia": "\U0001f1ed\U0001f1f7", "Cura\u00e7ao": "\U0001f1e8\U0001f1fc",
    "Czech Republic": "\U0001f1e8\U0001f1ff", "DR Congo": "\U0001f1e8\U0001f1e9", "Ecuador": "\U0001f1ea\U0001f1e8",
    "Egypt": "\U0001f1ea\U0001f1ec", "England": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "France": "\U0001f1eb\U0001f1f7", "Germany": "\U0001f1e9\U0001f1ea", "Ghana": "\U0001f1ec\U0001f1ed",
    "Haiti": "\U0001f1ed\U0001f1f9", "Iran": "\U0001f1ee\U0001f1f7", "Iraq": "\U0001f1ee\U0001f1f6",
    "Ivory Coast": "\U0001f1e8\U0001f1ee", "Japan": "\U0001f1ef\U0001f1f5", "Jordan": "\U0001f1ef\U0001f1f4",
    "Mexico": "\U0001f1f2\U0001f1fd", "Morocco": "\U0001f1f2\U0001f1e6", "Netherlands": "\U0001f1f3\U0001f1f1",
    "New Zealand": "\U0001f1f3\U0001f1ff", "Norway": "\U0001f1f3\U0001f1f4", "Panama": "\U0001f1f5\U0001f1e6",
    "Paraguay": "\U0001f1f5\U0001f1fe", "Portugal": "\U0001f1f5\U0001f1f9", "Qatar": "\U0001f1f6\U0001f1e6",
    "Saudi Arabia": "\U0001f1f8\U0001f1e6", "Scotland": "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "Senegal": "\U0001f1f8\U0001f1f3", "South Africa": "\U0001f1ff\U0001f1e6", "South Korea": "\U0001f1f0\U0001f1f7",
    "Spain": "\U0001f1ea\U0001f1f8", "Sweden": "\U0001f1f8\U0001f1ea", "Switzerland": "\U0001f1e8\U0001f1ed",
    "Tunisia": "\U0001f1f9\U0001f1f3", "Turkey": "\U0001f1f9\U0001f1f7", "United States": "\U0001f1fa\U0001f1f8",
    "Uruguay": "\U0001f1fa\U0001f1fe", "Uzbekistan": "\U0001f1fa\U0001f1ff",
}

FEATURES = [
    'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
    'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2',
    'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15', 'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2',
    'fifa_ranking', 'log_squad_value', 'avg_age',
    'rival_fifa_ranking', 'rival_log_squad_value', 'rival_avg_age'
]

st.set_page_config(page_title="Mundial 2026 Predictor", page_icon="\U0001f3c6", layout="wide")


@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model(MODEL_PATH)
    return m

@st.cache_data
def load_teams():
    df = pd.read_csv(DATA_PATH)
    td = df.drop_duplicates(subset="team", keep="first").set_index("team")
    return td, sorted(td.index.tolist())

@st.cache_data
def load_history():
    path = os.path.join(PROJECT_ROOT, "data", "prediction_history.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(entry):
    path = os.path.join(PROJECT_ROOT, "data", "prediction_history.json")
    history = load_history()
    history.insert(0, entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history[:50], f, ensure_ascii=False, indent=2)


model = load_model()
teams_df, TEAM_NAMES = load_teams()


def flag(name):
    return FLAGS.get(name, "\U0001f3f3\ufe0f")


def predict_xg(team_a, team_b):
    a, b = teams_df.loc[team_a], teams_df.loc[team_b]
    def build_row(name, opp, sd, od):
        log_self = np.log10(sd['squad_value'] + 1) if pd.notna(sd['squad_value']) and sd['squad_value'] > 0 else 0
        log_opp = np.log10(od['squad_value'] + 1) if pd.notna(od['squad_value']) and od['squad_value'] > 0 else 0
        return {
            'elo': sd['elo'], 'opponent_elo': od['elo'],
            'is_home': int(name in ["United States", "Canada", "Mexico"]),
            'tournament_num': 5, 'confed': sd['confed'], 'rival_confed': od['confed'],
            'gf_prom_5': sd['gf_prom_5'], 'gc_prom_5': sd['gc_prom_5'], 'elo_prom_5': sd['elo_prom_5'],
            'gf_prom_15': sd['gf_prom_15'], 'gc_prom_15': sd['gc_prom_15'], 'PCA_1': sd['PCA_1'], 'PCA_2': sd['PCA_2'],
            'rival_gf_prom_5': od['gf_prom_5'], 'rival_gc_prom_5': od['gc_prom_5'], 'rival_elo_prom_5': od['elo_prom_5'],
            'rival_gf_prom_15': od['gf_prom_15'], 'rival_gc_prom_15': od['gc_prom_15'], 'rival_PCA_1': od['PCA_1'], 'rival_PCA_2': od['PCA_2'],
            'fifa_ranking': sd['fifa_ranking'], 'log_squad_value': log_self, 'avg_age': sd['avg_age'],
            'rival_fifa_ranking': od['fifa_ranking'], 'rival_log_squad_value': log_opp, 'rival_avg_age': od['avg_age'],
        }
    rows = [build_row(team_a, team_b, a, b), build_row(team_b, team_a, b, a)]
    df_pred = pd.DataFrame(rows)
    df_pred['xg_estimated'] = model.predict(df_pred[FEATURES]).round(2)
    return df_pred['xg_estimated'].tolist()


class SimpleTeam:
    def __init__(self, name, elo):
        self.name, self.elo = name, elo
        self.points = self.dg = self.gf = 0

def simulate_match(xg1, xg2, elo1, elo2, name1, name2, n=200):
    t1, t2 = SimpleTeam(name1, elo1), SimpleTeam(name2, elo2)
    match = Match(t1, t2, xg1, xg2)
    for _ in range(n):
        match.simulate_match()
    return match.results


def render_bracket_html(knockouts):
    LABELS = {'first_round': '1/16', 'sweet16': '1/8', 'elite8': '1/4', 'semis': '1/2', 'final': 'Final'}
    KEYS = ['first_round', 'sweet16', 'elite8', 'semis', 'final']
    MH, G = 72, 8
    all_m = [m for r in knockouts for m in r['matches']]
    counts = [16, 8, 4, 2, 2]

    pos = []
    for ri, (off, n) in enumerate([(0, 16), (16, 8), (24, 4), (28, 2), (30, 2)]):
        for i in range(n):
            if ri == 0:
                pos.append(i * (MH + G))
            elif ri < 4:
                p = off - counts[ri - 1]
                t0, t1 = pos[p + 2 * i], pos[p + 2 * i + 1] + MH
                pos.append((t0 + t1) / 2 - MH / 2)
            else:
                # Final round: 2 matches (final + 3rd place), center between the 2 SF
                p = off - counts[ri - 1]
                center = (pos[p] + pos[p + 1] + MH) / 2
                pos.append(center - (MH + G) / 2 if i == 0 else center + (MH + G) / 2 - MH / 2)

    totalH = max(pos[30], pos[31]) + MH + 30
    LH = 28
    html = f'<div style="display:flex;gap:4px;min-height:{totalH + LH}px;padding:0 2px">'

    for ri, (off, n) in enumerate([(0, 16), (16, 8), (24, 4), (28, 2), (30, 2)]):
        html += f'<div style="flex:1;min-width:150px;position:relative;border-right:{1 if ri < 4 else 0}px solid #2a2a4a">'
        html += f'<div style="text-align:center;color:#fbbf24;font-weight:bold;font-size:.8rem;padding:4px 0;height:{LH}px">{LABELS[KEYS[ri]]}</div>'
        for i in range(n):
            m = all_m[off + i]
            if not m:
                continue
            w1, w2 = m['score1'] > m['score2'], m['score2'] > m['score1']
            top = pos[off + i] + LH + 4
            html += f'''<div style="position:absolute;top:{top}px;left:4px;right:4px;background:#252540;border-radius:8px;padding:6px 8px;border:1px solid #444;font-size:.75rem">
<div style="font-size:.95rem;font-weight:bold;color:#fff;text-align:center;margin-bottom:2px">{m['score1']} - {m['score2']}</div>
<div style="display:flex;align-items:center;gap:4px;padding:2px 0;{"color:#22c55e;font-weight:bold" if w1 else "opacity:.5"}">{m['flag1']}<span style="flex:1">{m['team1']}</span><span style="font-weight:bold;min-width:16px;text-align:right">{m['score1']}</span></div>
<div style="display:flex;align-items:center;gap:4px;padding:2px 0;{"color:#22c55e;font-weight:bold" if w2 else "opacity:.5"}">{m['flag2']}<span style="flex:1">{m['team2']}</span><span style="font-weight:bold;min-width:16px;text-align:right">{m['score2']}</span></div>
</div>'''
        html += '</div>'
    html += '</div>'
    return html


# ===== UI =====

st.title("\U0001f3c6 Mundial 2026 Predictor")
st.markdown("---")

tab1, tab2 = st.tabs(["\U0001f52e Predecir Partido", "\U0001f3c6 Simular Torneo"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        team_a = st.selectbox("Equipo A", TEAM_NAMES, index=TEAM_NAMES.index("Spain") if "Spain" in TEAM_NAMES else 0,
                              format_func=lambda x: f"{flag(x)} {x}")
    with col2:
        team_b = st.selectbox("Equipo B", TEAM_NAMES, index=TEAM_NAMES.index("France") if "France" in TEAM_NAMES else 1,
                              format_func=lambda x: f"{flag(x)} {x}")

    sims = st.slider("N\u00famero de simulaciones", 50, 500, 200, 50)

    if st.button("\U0001f52e Predecir", use_container_width=True, type="primary"):
        if team_a == team_b:
            st.error("Selecciona dos equipos diferentes")
        else:
            with st.spinner(f"Simulando {sims} veces..."):
                xgs = predict_xg(team_a, team_b)
                xg_a, xg_b = float(xgs[0]), float(xgs[1])
                elo_a = float(teams_df.loc[team_a, 'elo'])
                elo_b = float(teams_df.loc[team_b, 'elo'])
                results = simulate_match(xg_a, xg_b, elo_a, elo_b, team_a, team_b, n=sims)

                total = sum(results.values())
                win1 = sum(v for (g1, g2), v in results.items() if g1 > g2)
                draw = sum(v for (g1, g2), v in results.items() if g1 == g2)
                win2 = sum(v for (g1, g2), v in results.items() if g2 > g1)
                p1, pd, p2 = win1 / total * 100, draw / total * 100, win2 / total * 100
                sorted_scores = sorted(results.items(), key=lambda x: -x[1])

            st.markdown(f"### Resultados tras {sims} simulaciones")

            col_a, col_d, col_b = st.columns([p1, pd, p2])
            with col_a:
                st.markdown(f"<div style='background:#3b82f6;border-radius:8px;padding:10px;text-align:center;color:#fff'><strong>{team_a}</strong><br><span style='font-size:1.5rem'>{p1:.1f}%</span></div>", unsafe_allow_html=True)
            with col_d:
                st.markdown(f"<div style='background:#6b7280;border-radius:8px;padding:10px;text-align:center;color:#fff'><strong>Empate</strong><br><span style='font-size:1.5rem'>{pd:.1f}%</span></div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div style='background:#ef4444;border-radius:8px;padding:10px;text-align:center;color:#fff'><strong>{team_b}</strong><br><span style='font-size:1.5rem'>{p2:.1f}%</span></div>", unsafe_allow_html=True)

            g1, g2 = st.columns(2)
            g1.metric(f"xG {team_a}", f"{xg_a:.2f}")
            g2.metric(f"xG {team_b}", f"{xg_b:.2f}")

            st.markdown("#### Marcadores m\u00e1s probables")
            cols = st.columns(5)
            for i, ((s1, s2), v) in enumerate(sorted_scores[:5]):
                with cols[i]:
                    st.markdown(f"<div style='background:#252540;border-radius:8px;padding:8px;text-align:center'><div style='font-size:1.2rem;font-weight:bold'>{s1}-{s2}</div><div style='color:#fbbf24'>{v/total*100:.1f}%</div></div>", unsafe_allow_html=True)

            save_history({"type": "match", "date": datetime.now().strftime("%d/%m %H:%M"),
                          "team_a": team_a, "team_b": team_b,
                          "win1_pct": round(p1, 1), "draw_pct": round(pd, 1), "win2_pct": round(p2, 1)})

with tab2:
    if st.button("\U0001f3c6 Simular Torneo Completo", use_container_width=True, type="primary"):
        with st.spinner("Simulando torneo completo (105 partidos)..."):
            t = Tournament(name="misterclaude", model=model)
            t.simulate_tournament()
            t.export_results()

            st.markdown("### Fase de Grupos")
            best_third_names = {third.name for third in t.thirds}
            cols = st.columns(4)
            for i, group in enumerate(t.GROUPS):
                order = t.group_orders[i]
                with cols[i % 4]:
                    st.markdown(f"**Grupo {group.letter}**")
                    data = []
                    for j, team in enumerate(order):
                        cls = "qualified" if j < 2 else ("third" if j == 2 and team.name in best_third_names else "")
                        data.append({"#": j + 1, "Equipo": f"{flag(team.name)} {team.name}",
                                     "Pts": team.points, "DG": team.dg, "GF": team.gf, "": cls})
                    st.dataframe(data, column_config={"": None}, hide_index=True, use_container_width=True)

            st.markdown("### Fase Eliminatoria")
            ko = t.knockouts
            ROUNDS = ["first_round", "sweet16", "elite8", "semis", "final"]
            idx = 0
            knockouts_data = []
            for rnd_name in ROUNDS:
                match_count = {"first_round": 16, "sweet16": 8, "elite8": 4, "semis": 2, "final": 2}[rnd_name]
                round_matches = ko.results[idx:idx + match_count]
                idx += match_count
                matches = []
                for m in round_matches:
                    matches.append({
                        "team1": m["Team_1"], "flag1": flag(m["Team_1"]),
                        "team2": m["Team_2"], "flag2": flag(m["Team_2"]),
                        "score1": m["Score_1"], "score2": m["Score_2"],
                    })
                knockouts_data.append({"round": rnd_name, "matches": matches})

            bracket_html = render_bracket_html(knockouts_data)
            st.components.v1.html(f"<div style='background:#1a1a2e;padding:10px;border-radius:12px'>{bracket_html}</div>", height=900, scrolling=True)

            winner_name = t.winner.name if t.winner else "?"
            st.balloons()
            st.markdown(f"## \U0001f3c6 {flag(winner_name)} **{winner_name}** es el campe\u00f3n del Mundial 2026!")

            save_history({"type": "tournament", "date": datetime.now().strftime("%d/%m %H:%M"), "winner": winner_name})

with st.sidebar:
    st.markdown(f"### \U0001f4cb Historial")
    history = load_history()
    if history:
        for h in history[:10]:
            if h.get("type") == "match":
                st.markdown(f"**{h.get('team_a', '?')}** vs **{h.get('team_b', '?')}**  \n{h.get('win1_pct', 0):.0f}% / {h.get('draw_pct', 0):.0f}% / {h.get('win2_pct', 0):.0f}%  \n`{h.get('date', '')}`")
                st.divider()
            elif h.get("type") == "tournament":
                st.markdown(f"\U0001f3c6 **{h.get('winner', '?')}** campe\u00f3n  \n`{h.get('date', '')}`")
                st.divider()
    else:
        st.markdown("Sin predicciones a\u00fan")

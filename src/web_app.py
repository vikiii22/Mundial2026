import os, sys, json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from src.clases_simulacion import Match, Tournament

MODEL_PATH = os.path.join(PROJECT_ROOT, "data/ai_models/xg_model_misterclaude.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/ai_models/xg_preds_J1_misterclaude_complete.csv")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "data/prediction_history.json")

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

df_full = pd.read_csv(DATA_PATH)
teams_df = df_full.drop_duplicates(subset="team", keep="first").set_index("team")
TEAM_NAMES = sorted(teams_df.index.tolist())

FEATURES = [
    'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
    'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2',
    'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15', 'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2',
    'fifa_ranking', 'log_squad_value', 'avg_age',
    'rival_fifa_ranking', 'rival_log_squad_value', 'rival_avg_age'
]

app = FastAPI(title="Mundial 2026 Predictor")


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[:50], f, ensure_ascii=False, indent=2)


def predict_xg(team_a: str, team_b: str):
    a = teams_df.loc[team_a]
    b = teams_df.loc[team_b]

    def build_row(name, opp, sd, od):
        log_self = np.log10(sd['squad_value'] + 1) if pd.notna(sd['squad_value']) and sd['squad_value'] > 0 else 0
        log_opp = np.log10(od['squad_value'] + 1) if pd.notna(od['squad_value']) and od['squad_value'] > 0 else 0
        return {
            'elo': sd['elo'], 'opponent_elo': od['elo'],
            'is_home': int(name in ["United States", "Canada", "Mexico"]),
            'tournament_num': 5,
            'confed': sd['confed'], 'rival_confed': od['confed'],
            'gf_prom_5': sd['gf_prom_5'], 'gc_prom_5': sd['gc_prom_5'],
            'elo_prom_5': sd['elo_prom_5'],
            'gf_prom_15': sd['gf_prom_15'], 'gc_prom_15': sd['gc_prom_15'],
            'PCA_1': sd['PCA_1'], 'PCA_2': sd['PCA_2'],
            'rival_gf_prom_5': od['gf_prom_5'], 'rival_gc_prom_5': od['gc_prom_5'],
            'rival_elo_prom_5': od['elo_prom_5'],
            'rival_gf_prom_15': od['gf_prom_15'], 'rival_gc_prom_15': od['gc_prom_15'],
            'rival_PCA_1': od['PCA_1'], 'rival_PCA_2': od['PCA_2'],
            'fifa_ranking': sd['fifa_ranking'],
            'log_squad_value': log_self,
            'avg_age': sd['avg_age'],
            'rival_fifa_ranking': od['fifa_ranking'],
            'rival_log_squad_value': log_opp,
            'rival_avg_age': od['avg_age'],
        }

    rows = [build_row(team_a, team_b, a, b), build_row(team_b, team_a, b, a)]
    df_pred = pd.DataFrame(rows)
    df_pred['xg_estimated'] = model.predict(df_pred[FEATURES]).round(2)
    return df_pred['xg_estimated'].tolist()


class SimpleTeam:
    def __init__(self, name, elo):
        self.name = name
        self.elo = elo
        self.points = 0
        self.dg = 0
        self.gf = 0


def simulate_match(xg1, xg2, elo1, elo2, name1, name2, n=200):
    t1 = SimpleTeam(name1, elo1)
    t2 = SimpleTeam(name2, elo2)
    match = Match(t1, t2, xg1, xg2)
    for _ in range(n):
        match.simulate_match()
    return match.results


def build_html(teams, history=None):
    options = "".join(f'<option value="{t}">{t}</option>' for t in teams)
    hist_html = ""
    if history:
        for h in history[:10]:
            if h.get("type") == "match":
                hist_html += f'<div class="hist-item"><span class="hist-teams">{h["team_a"]} vs {h["team_b"]}</span><span class="hist-pct">{h["win1_pct"]:.0f}% / {h["draw_pct"]:.0f}% / {h["win2_pct"]:.0f}%</span><span class="hist-date">{h["date"]}</span></div>'
            elif h.get("type") == "tournament":
                hist_html += f'<div class="hist-item"><span class="hist-teams">\U0001f3c6 Torneo: {h["winner"]}</span><span class="hist-date">{h["date"]}</span></div>'
    hist_section = f'<div class="card" id="historyCard"><h2 style="color:#fbbf24;margin-bottom:16px;font-size:1.1rem">\U0001f4cb Historial</h2><div class="hist-list">{hist_html or "<div style=color:#666;text-align:center;padding:10px>Sin predicciones aun</div>"}</div></div>' if hist_html or not history else ""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mundial 2026 - Predictor</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; display: flex; justify-content: center; padding: 20px; }}
  .container {{ max-width: 750px; width: 100%; }}
  h1 {{ text-align: center; font-size: 1.8rem; margin: 20px 0; color: #fff; }}
  h1 span {{ color: #fbbf24; }}
  .card {{ background: #1a1a2e; border-radius: 16px; padding: 30px; margin-bottom: 20px; border: 1px solid #2a2a4a; }}
  .team-row {{ display: flex; gap: 16px; align-items: center; justify-content: center; flex-wrap: wrap; }}
  .team-row select {{ flex: 1; min-width: 160px; padding: 12px 16px; border-radius: 10px; border: 1px solid #3a3a5a; background: #252540; color: #e0e0e0; font-size: 1rem; cursor: pointer; }}
  .team-row select:focus {{ outline: none; border-color: #fbbf24; }}
  .vs {{ font-size: 1.3rem; font-weight: bold; color: #fbbf24; padding: 0 8px; }}
  .sim-slider {{ display: flex; align-items: center; gap: 12px; justify-content: center; margin: 20px 0; }}
  .sim-slider label {{ color: #aaa; }}
  .sim-slider input[type=range] {{ width: 200px; accent-color: #fbbf24; }}
  .sim-slider .val {{ color: #fbbf24; font-weight: bold; min-width: 40px; text-align: center; }}
  .btn {{ display: block; width: 100%; padding: 14px; border: none; border-radius: 10px; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: transform .15s; }}
  .btn:hover {{ transform: translateY(-1px); }}
  .btn:active {{ transform: translateY(0); }}
  .btn:disabled {{ opacity: .5; cursor: wait; }}
  .btn-primary {{ background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #000; }}
  .btn-secondary {{ background: linear-gradient(135deg, #6366f1, #4f46e5); color: #fff; margin-top: 12px; }}
  .divider {{ height: 1px; background: #2a2a4a; margin: 24px 0; }}
  .results {{ margin-top: 20px; }}
  .results h2 {{ text-align: center; margin-bottom: 20px; color: #fbbf24; font-size: 1.2rem; }}
  .prob-bar {{ display: flex; height: 36px; border-radius: 8px; overflow: hidden; margin: 16px 0; font-size: .8rem; font-weight: bold; }}
  .prob-a {{ background: #3b82f6; display: flex; align-items: center; justify-content: center; transition: width .5s; min-width: fit-content; padding: 0 6px; white-space: nowrap; }}
  .prob-draw {{ background: #6b7280; display: flex; align-items: center; justify-content: center; transition: width .5s; min-width: fit-content; padding: 0 6px; white-space: nowrap; }}
  .prob-b {{ background: #ef4444; display: flex; align-items: center; justify-content: center; transition: width .5s; min-width: fit-content; padding: 0 6px; white-space: nowrap; }}
  .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
  .stat-card {{ background: #252540; border-radius: 10px; padding: 14px; text-align: center; }}
  .stat-card .label {{ color: #888; font-size: .8rem; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 1.5rem; font-weight: bold; margin-top: 4px; }}
  .scorelines {{ margin-top: 16px; }}
  .scorelines h3 {{ color: #aaa; font-size: .9rem; margin-bottom: 10px; }}
  .score-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 8px; }}
  .score-item {{ background: #252540; border-radius: 8px; padding: 10px; text-align: center; }}
  .score-item .sc {{ font-weight: bold; font-size: 1.1rem; }}
  .score-item .pct {{ color: #fbbf24; font-size: .85rem; }}
  .loading {{ text-align: center; padding: 30px; color: #888; }}
  .error {{ text-align: center; padding: 20px; color: #f87171; }}
  .hist-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .hist-item {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #252540; border-radius: 8px; font-size: .85rem; flex-wrap: wrap; gap: 6px; }}
  .hist-teams {{ font-weight: bold; }}
  .hist-pct {{ color: #fbbf24; }}
  .hist-date {{ color: #666; font-size: .75rem; }}
  .tournament-result {{ text-align: center; padding: 20px; }}
  .tournament-result .winner-name {{ font-size: 1.8rem; font-weight: bold; color: #fbbf24; margin: 10px 0; }}
  .tournament-result .sub {{ color: #888; }}
  @media (max-width: 500px) {{ .team-row select {{ min-width: 120px; font-size: .9rem; }} .stats {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>\U0001f3c6 Mundial 2026 <span>Predictor</span></h1>

  <div class="card">
    <form id="predForm" onsubmit="return predict(event)">
      <div class="team-row">
        <select name="team_a" id="team_a">{options}</select>
        <span class="vs">vs</span>
        <select name="team_b" id="team_b">{options}</select>
      </div>
      <div class="sim-slider">
        <label>Simulaciones:</label>
        <input type="range" name="simulations" id="simulations" min="50" max="500" value="200" step="50" oninput="document.getElementById('simVal').textContent=this.value">
        <span class="val" id="simVal">200</span>
      </div>
      <button class="btn btn-primary" id="predBtn" type="submit">\U0001f52e Predecir</button>
    </form>
    <div class="divider"></div>
    <button class="btn btn-secondary" id="tournBtn" onclick="simulateTournament()">\U0001f3c6 Simular Torneo Completo</button>
  </div>

  <div id="results"></div>

  {hist_section}
</div>

<script>
async function predict(e) {{
  e.preventDefault();
  const btn = document.getElementById('predBtn');
  const out = document.getElementById('results');
  btn.disabled = true; btn.textContent = '\u23f3 Analizando...';
  out.innerHTML = '<div class="loading">Simulando partido...</div>';
  const fd = new FormData(document.getElementById('predForm'));
  try {{
    const r = await fetch('/predict', {{ method: 'POST', body: fd }});
    const d = await r.json();
    if (d.error) {{ out.innerHTML = '<div class="error">' + d.error + '</div>'; return; }}
    out.innerHTML = renderResults(d);
    refreshHistory();
  }} catch(e) {{ out.innerHTML = '<div class="error">Error de conexion</div>'; }}
  finally {{ btn.disabled = false; btn.textContent = '\U0001f52e Predecir'; }}
}}

function renderResults(d) {{
  const p1 = d.win1_pct, pd = d.draw_pct, p2 = d.win2_pct;
  const t1 = d.team_a, t2 = d.team_b;
  const top5 = d.top_scorelines.slice(0, 5);
  const bar1 = p1 > 8 ? t1 + ' ' + p1.toFixed(1) + '%' : '';
  const bard = pd > 8 ? 'Emp ' + pd.toFixed(1) + '%' : '';
  const bar2 = p2 > 8 ? t2 + ' ' + p2.toFixed(1) + '%' : '';
  const sl = top5.map(function(s) {{
    return '<div class="score-item"><div class="sc">' + s.score + '</div><div class="pct">' + s.pct.toFixed(1) + '%</div></div>';
  }}).join('');
  return '<div class="card results">' +
    '<h2>Resultados tras ' + d.simulations + ' simulaciones</h2>' +
    '<div class="prob-bar">' +
      '<div class="prob-a" style="width:' + p1 + '%">' + bar1 + '</div>' +
      '<div class="prob-draw" style="width:' + pd + '%">' + bard + '</div>' +
      '<div class="prob-b" style="width:' + p2 + '%">' + bar2 + '</div>' +
    '</div>' +
    '<div style="display:flex;justify-content:space-between;font-size:.8rem;color:#888;margin-top:-8px;padding:0 4px">' +
      '<span>' + t1 + ': <strong style="color:#60a5fa">' + p1.toFixed(1) + '%</strong></span>' +
      '<span>Empate: <strong style="color:#aaa">' + pd.toFixed(1) + '%</strong></span>' +
      '<span>' + t2 + ': <strong style="color:#f87171">' + p2.toFixed(1) + '%</strong></span>' +
    '</div>' +
    '<div class="stats">' +
      '<div class="stat-card"><div class="label">' + t1 + ' - xG</div><div class="value" style="color:#60a5fa">' + d.xg_a.toFixed(2) + '</div></div>' +
      '<div class="stat-card"><div class="label">' + t2 + ' - xG</div><div class="value" style="color:#f87171">' + d.xg_b.toFixed(2) + '</div></div>' +
    '</div>' +
    '<div class="scorelines"><h3>\u25b6 Marcadores mas probables</h3>' +
      '<div class="score-grid">' + sl + '</div>' +
    '</div></div>';
}}

async function simulateTournament() {{
  const btn = document.getElementById('tournBtn');
  const out = document.getElementById('results');
  btn.disabled = true; btn.textContent = '\u23f3 Simulando torneo (105 partidos)...';
  out.innerHTML = '<div class="loading"><div style="font-size:1.2rem;margin-bottom:8px">\U0001f3c6 Simulando torneo completo</div><div style="color:#666">Esto puede tomar hasta 30 segundos...</div></div>';
  try {{
    const r = await fetch('/simulate_tournament', {{ method: 'POST' }});
    const d = await r.json();
    if (d.error) {{ out.innerHTML = '<div class="error">' + d.error + '</div>'; return; }}
    let html = '<div class="card tournament-result">' +
      '<h2>\U0001f3c6 Resultado del Torneo</h2>' +
      '<div class="winner-name">' + d.winner + '</div>' +
      '<div class="sub">Campeon del Mundial 2026</div>' +
      '<div class="stats" style="margin-top:16px">' +
        '<div class="stat-card"><div class="label">Partidos</div><div class="value">' + d.total_matches + '</div></div>' +
        '<div class="stat-card"><div class="label">Goles totales</div><div class="value">' + d.total_goals + '</div></div>' +
      '</div>';
    if (d.top_scorer) {{
      html += '<div style="margin-top:12px;color:#888">\u26bd Maximo goleador: <strong style="color:#fbbf24">' + d.top_scorer + '</strong></div>';
    }}
    html += '</div>';
    out.innerHTML = html;
    refreshHistory();
  }} catch(e) {{ out.innerHTML = '<div class="error">Error: ' + e.message + '</div>'; }}
  finally {{ btn.disabled = false; btn.textContent = '\U0001f3c6 Simular Torneo Completo'; }}
}}

async function refreshHistory() {{
  try {{
    const r = await fetch('/history');
    const d = await r.json();
    let html = '';
    for (const h of d) {{
      if (h.type === 'match') {{
        html += '<div class="hist-item"><span class="hist-teams">' + h.team_a + ' vs ' + h.team_b + '</span><span class="hist-pct">' + h.win1_pct.toFixed(0) + '% / ' + h.draw_pct.toFixed(0) + '% / ' + h.win2_pct.toFixed(0) + '%</span><span class="hist-date">' + h.date + '</span></div>';
      }} else if (h.type === 'tournament') {{
        html += '<div class="hist-item"><span class="hist-teams">\U0001f3c6 Torneo: ' + h.winner + '</span><span class="hist-date">' + h.date + '</span></div>';
      }}
    }}
    const card = document.getElementById('historyCard');
    if (card && html) {{
      card.querySelector('.hist-list').innerHTML = html;
    }} else if (html) {{
      const histSection = '<div class="card" id="historyCard"><h2 style="color:#fbbf24;margin-bottom:16px;font-size:1.1rem">\U0001f4cb Historial</h2><div class="hist-list">' + html + '</div></div>';
      document.querySelector('.container').insertAdjacentHTML('beforeend', histSection);
    }}
  }} catch(e) {{}}
}}
</script>
</body>
</html>"""
    return html


@app.get("/", response_class=HTMLResponse)
def index():
    return build_html(TEAM_NAMES, load_history())


@app.post("/predict")
def predict(team_a: str = Form(...), team_b: str = Form(...), simulations: int = Form(200)):
    if team_a == team_b:
        return {"error": "Selecciona dos equipos diferentes"}
    if team_a not in teams_df.index or team_b not in teams_df.index:
        return {"error": "Equipo no valido"}

    xgs = predict_xg(team_a, team_b)
    xg_a, xg_b = float(xgs[0]), float(xgs[1])
    elo_a = float(teams_df.loc[team_a, 'elo'])
    elo_b = float(teams_df.loc[team_b, 'elo'])

    results = simulate_match(xg_a, xg_b, elo_a, elo_b, team_a, team_b, n=simulations)

    total = sum(results.values())
    win1 = sum(v for (g1, g2), v in results.items() if g1 > g2)
    draw = sum(v for (g1, g2), v in results.items() if g1 == g2)
    win2 = sum(v for (g1, g2), v in results.items() if g2 > g1)

    sorted_scores = sorted(results.items(), key=lambda x: -x[1])
    top_scorelines = [
        {"score": f"{s[0]}-{s[1]}", "pct": round(v / total * 100, 2)}
        for (s, v) in sorted_scores[:10]
    ]

    entry = {
        "type": "match",
        "date": datetime.now().strftime("%d/%m %H:%M"),
        "team_a": team_a,
        "team_b": team_b,
        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),
        "win1_pct": round(win1 / total * 100, 2),
        "draw_pct": round(draw / total * 100, 2),
        "win2_pct": round(win2 / total * 100, 2),
        "simulations": simulations,
    }
    save_history(entry)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),
        "simulations": simulations,
        "win1_pct": round(win1 / total * 100, 2),
        "draw_pct": round(draw / total * 100, 2),
        "win2_pct": round(win2 / total * 100, 2),
        "top_scorelines": top_scorelines,
    }


@app.post("/simulate_tournament")
def simulate_tournament():
    try:
        t = Tournament(name="misterclaude", model=model)
        t.simulate_tournament()
        df = t.export_results()
        winner_name = t.winner.name if t.winner else "Desconocido"

        total_matches = len(df)
        total_goals = int(df["Score_1"].sum() + df["Score_2"].sum())

        entry = {
            "type": "tournament",
            "date": datetime.now().strftime("%d/%m %H:%M"),
            "winner": winner_name,
            "total_matches": total_matches,
            "total_goals": total_goals,
        }
        save_history(entry)

        return {
            "winner": winner_name,
            "total_matches": total_matches,
            "total_goals": total_goals,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/history")
def get_history():
    return load_history()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

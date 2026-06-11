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

def flag(name):
    return FLAGS.get(name, "\U0001f3f3\ufe0f")

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


def build_html(teams, history=None):
    options = "".join(f'<option value="{t}">{flag(t)} {t}</option>' for t in teams)
    hist_html = ""
    if history:
        for h in history[:10]:
            if h.get("type") == "match":
                hist_html += f'<div class="hist-item"><span class="hist-teams">{flag(h["team_a"])} {h["team_a"]} vs {flag(h["team_b"])} {h["team_b"]}</span><span class="hist-pct">{h["win1_pct"]:.0f}% / {h["draw_pct"]:.0f}% / {h["win2_pct"]:.0f}%</span><span class="hist-date">{h["date"]}</span></div>'
            elif h.get("type") == "tournament":
                hist_html += f'<div class="hist-item"><span class="hist-teams">\U0001f3c6 Torneo: {flag(h["winner"])} {h["winner"]}</span><span class="hist-date">{h["date"]}</span></div>'
    hist_section = f'<div class="card" id="historyCard"><h2 style="color:#fbbf24;margin-bottom:16px;font-size:1.1rem">\U0001f4cb Historial</h2><div class="hist-list">{hist_html or "<div style=color:#666;text-align:center;padding:10px>Sin predicciones aun</div>"}</div></div>' if (history and hist_html) else ""

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mundial 2026 - Predictor</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f0f1a; color:#e0e0e0; min-height:100vh; display:flex; justify-content:center; padding:20px; }}
  .container {{ max-width:950px; width:100%; }}
  h1 {{ text-align:center; font-size:1.8rem; margin:20px 0; color:#fff; }}
  h1 span {{ color:#fbbf24; }}
  .card {{ background:#1a1a2e; border-radius:16px; padding:24px; margin-bottom:20px; border:1px solid #2a2a4a; }}
  .team-row {{ display:flex; gap:12px; align-items:center; justify-content:center; flex-wrap:wrap; }}
  .team-row select {{ flex:1; min-width:150px; padding:10px 14px; border-radius:10px; border:1px solid #3a3a5a; background:#252540; color:#e0e0e0; font-size:.95rem; cursor:pointer; }}
  .team-row select:focus {{ outline:none; border-color:#fbbf24; }}
  .vs {{ font-size:1.2rem; font-weight:bold; color:#fbbf24; padding:0 6px; }}
  .sim-slider {{ display:flex; align-items:center; gap:10px; justify-content:center; margin:16px 0; }}
  .sim-slider label {{ color:#aaa; font-size:.9rem; }}
  .sim-slider input[type=range] {{ width:180px; accent-color:#fbbf24; }}
  .sim-slider .val {{ color:#fbbf24; font-weight:bold; min-width:35px; text-align:center; }}
  .btn {{ display:block; width:100%; padding:12px; border:none; border-radius:10px; font-size:1rem; font-weight:bold; cursor:pointer; transition:transform .15s; }}
  .btn:hover {{ transform:translateY(-1px); }}
  .btn:active {{ transform:translateY(0); }}
  .btn:disabled {{ opacity:.5; cursor:wait; }}
  .btn-primary {{ background:linear-gradient(135deg,#fbbf24,#f59e0b); color:#000; }}
  .btn-secondary {{ background:linear-gradient(135deg,#6366f1,#4f46e5); color:#fff; margin-top:10px; }}
  .divider {{ height:1px; background:#2a2a4a; margin:20px 0; }}
  .results {{ margin-top:16px; }}
  .results h2 {{ text-align:center; margin-bottom:16px; color:#fbbf24; font-size:1.1rem; }}
  .prob-bar {{ display:flex; height:32px; border-radius:8px; overflow:hidden; margin:12px 0; font-size:.75rem; font-weight:bold; }}
  .prob-a {{ background:#3b82f6; display:flex; align-items:center; justify-content:center; transition:width .5s; min-width:fit-content; padding:0 6px; white-space:nowrap; }}
  .prob-draw {{ background:#6b7280; display:flex; align-items:center; justify-content:center; transition:width .5s; min-width:fit-content; padding:0 6px; white-space:nowrap; }}
  .prob-b {{ background:#ef4444; display:flex; align-items:center; justify-content:center; transition:width .5s; min-width:fit-content; padding:0 6px; white-space:nowrap; }}
  .stats {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0; }}
  .stat-card {{ background:#252540; border-radius:10px; padding:12px; text-align:center; }}
  .stat-card .label {{ color:#888; font-size:.75rem; text-transform:uppercase; }}
  .stat-card .value {{ font-size:1.3rem; font-weight:bold; margin-top:4px; }}
  .scorelines {{ margin-top:12px; }}
  .scorelines h3 {{ color:#aaa; font-size:.85rem; margin-bottom:8px; }}
  .score-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(100px,1fr)); gap:6px; }}
  .score-item {{ background:#252540; border-radius:8px; padding:8px; text-align:center; }}
  .score-item .sc {{ font-weight:bold; font-size:1rem; }}
  .score-item .pct {{ color:#fbbf24; font-size:.8rem; }}
  .loading {{ text-align:center; padding:30px; color:#888; }}
  .error {{ text-align:center; padding:20px; color:#f87171; }}
  .hist-list {{ display:flex; flex-direction:column; gap:6px; }}
  .hist-item {{ display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:#252540; border-radius:8px; font-size:.8rem; flex-wrap:wrap; gap:4px; }}
  .hist-teams {{ font-weight:bold; }}
  .hist-pct {{ color:#fbbf24; }}
  .hist-date {{ color:#666; font-size:.7rem; }}

  .group-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; margin-top:12px; }}
  .group-card {{ background:#252540; border-radius:10px; padding:12px; border:1px solid #333; }}
  .group-card h3 {{ color:#fbbf24; font-size:.9rem; margin-bottom:8px; text-align:center; }}
  .group-table {{ width:100%; border-collapse:collapse; font-size:.8rem; }}
  .group-table th {{ color:#888; padding:4px 2px; text-align:left; font-weight:normal; font-size:.7rem; }}
  .group-table td {{ padding:4px 2px; border-top:1px solid #333; }}
  .group-table .pos {{ width:18px; color:#666; text-align:center; }}
  .group-table .name {{ }}
  .group-table .pts {{ width:28px; text-align:center; font-weight:bold; }}
  .group-table .dg {{ width:28px; text-align:center; color:#888; }}
  .group-table .gf {{ width:28px; text-align:center; color:#888; }}
  .group-table .qualified-1 {{ border-left:3px solid #22c55e; }}
  .group-table .qualified-2 {{ border-left:3px solid #22c55e; }}
  .group-table .qualified-3 {{ border-left:3px solid #eab308; }}
  .group-table .eliminated {{ opacity:.5; }}

  .bracket-wrap {{ overflow-x:auto; padding:10px 0; }}

  .tournament-winner {{ text-align:center; padding:20px; }}
  .tournament-winner .trophy {{ font-size:3rem; }}
  .tournament-winner .wname {{ font-size:2rem; font-weight:bold; color:#fbbf24; margin:8px 0; }}
  .tournament-winner .sub {{ color:#888; }}

  @media (max-width:600px) {{ .group-grid {{ grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); }} .match-card {{ min-width:140px; flex:1 0 140px; }} .team-row select {{ min-width:110px; font-size:.85rem; }} }}
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
      <button class="btn btn-primary" id="predBtn" type="submit">\U0001f52e Predecir Partido</button>
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
  const btn=document.getElementById('predBtn'), out=document.getElementById('results');
  btn.disabled=true; btn.textContent='\u23f3 Analizando...';
  out.innerHTML='<div class="loading">Simulando partido...</div>';
  try {{
    const r=await fetch('/predict',{{method:'POST',body:new FormData(document.getElementById('predForm'))}});
    const d=await r.json();
    if(d.error){{out.innerHTML='<div class="error">'+d.error+'</div>';return;}}
    out.innerHTML=renderMatch(d);
    refreshHistory();
  }}catch(e){{out.innerHTML='<div class="error">Error de conexion</div>';}}
  finally{{btn.disabled=false;btn.textContent='\U0001f52e Predecir Partido';}}
}}

function renderMatch(d) {{
  const p1=d.win1_pct,pd=d.draw_pct,p2=d.win2_pct,t1=d.team_a,t2=d.team_b;
  const top5=d.top_scorelines.slice(0,5);
  const b1=p1>8?t1+' '+p1.toFixed(1)+'%':'',bd=pd>8?'Emp '+pd.toFixed(1)+'%':'',b2=p2>8?t2+' '+p2.toFixed(1)+'%':'';
  const sl=top5.map(function(s){{return '<div class="score-item"><div class="sc">'+s.score+'</div><div class="pct">'+s.pct.toFixed(1)+'%</div></div>';}}).join('');
  return '<div class="card results"><h2>Resultados tras '+d.simulations+' simulaciones</h2>'+
    '<div class="prob-bar"><div class="prob-a" style="width:'+p1+'%">'+b1+'</div><div class="prob-draw" style="width:'+pd+'%">'+bd+'</div><div class="prob-b" style="width:'+p2+'%">'+b2+'</div></div>'+
    '<div style="display:flex;justify-content:space-between;font-size:.75rem;color:#888;margin-top:-6px;padding:0 4px">'+
      '<span>'+t1+': <strong style="color:#60a5fa">'+p1.toFixed(1)+'%</strong></span>'+
      '<span>Empate: <strong style="color:#aaa">'+pd.toFixed(1)+'%</strong></span>'+
      '<span>'+t2+': <strong style="color:#f87171">'+p2.toFixed(1)+'%</strong></span></div>'+
    '<div class="stats">'+
      '<div class="stat-card"><div class="label">'+t1+' - xG</div><div class="value" style="color:#60a5fa">'+d.xg_a.toFixed(2)+'</div></div>'+
      '<div class="stat-card"><div class="label">'+t2+' - xG</div><div class="value" style="color:#f87171">'+d.xg_b.toFixed(2)+'</div></div></div>'+
    '<div class="scorelines"><h3>\u25b6 Marcadores mas probables</h3><div class="score-grid">'+sl+'</div></div></div>';
}}

function renderBracket(knockouts) {{
  const LABELS={{'first_round':'1/16','sweet16':'1/8','elite8':'1/4','semis':'1/2','final':'Final'}};
  const KEYS=['first_round','sweet16','elite8','semis','final'];
  const MH=72, G=8;
  const all=[].concat(...knockouts.map(r=>r.matches));
  const counts=[16,8,4,2,2];

  // Calculate absolute top positions for all 32 matches
  const pos=[];
  for(let ri=0,off=0;ri<5;ri++){{
    for(let i=0;i<counts[ri];i++){{
      if(ri===0) pos.push(i*(MH+G));
      else {{
        const p=off-counts[ri-1];
        const t0=pos[p+2*i], t1=pos[p+2*i+1]+MH;
        pos.push((t0+t1)/2-MH/2);
      }}
    }}
    off+=counts[ri];
  }}

  const totalH=pos[30]+MH+30, LH=28; // label height
  let html='<div style="display:flex;gap:4px;min-height:'+(totalH+LH)+'px;padding:0 2px">';
  for(let ri=0,off=0;ri<5;ri++){{
    const n=counts[ri];
    html+='<div style="flex:1;min-width:150px;position:relative;border-right:'+(ri<4?'1px solid #2a2a4a':'none')+'">'+
      '<div style="text-align:center;color:#fbbf24;font-weight:bold;font-size:.8rem;padding:4px 0;height:'+LH+'px">'+(LABELS[KEYS[ri]]||KEYS[ri])+'</div>';
    for(let i=0;i<n;i++){{
      const m=all[off+i];
      if(!m) continue;
      const w1=m.score1>m.score2, w2=m.score2>m.score1;
      html+='<div style="position:absolute;top:'+(pos[off+i]+LH+4)+'px;left:4px;right:4px;background:#252540;border-radius:8px;padding:6px 8px;border:1px solid #444;font-size:.75rem">'+
        '<div style="font-size:.95rem;font-weight:bold;color:#fff;text-align:center;margin-bottom:2px">'+m.score1+' - '+m.score2+'</div>'+
        '<div style="display:flex;align-items:center;gap:4px;padding:2px 0;'+(w1?'color:#22c55e;font-weight:bold':'opacity:.5')+'">'+m.flag1+'<span style="flex:1">'+m.team1+'</span><span style="font-weight:bold;min-width:16px;text-align:right">'+m.score1+'</span></div>'+
        '<div style="display:flex;align-items:center;gap:4px;padding:2px 0;'+(w2?'color:#22c55e;font-weight:bold':'opacity:.5')+'">'+m.flag2+'<span style="flex:1">'+m.team2+'</span><span style="font-weight:bold;min-width:16px;text-align:right">'+m.score2+'</span></div>'+
        '</div>';
    }}
    html+='</div>';
    off+=n;
  }}
  html+='</div>';
  return html;
}}

async function simulateTournament() {{
  const btn=document.getElementById('tournBtn'), out=document.getElementById('results');
  btn.disabled=true; btn.textContent='\u23f3 Simulando torneo...';
  out.innerHTML='<div class="loading"><div style="font-size:1.1rem;margin-bottom:8px">\U0001f3c6 Simulando torneo completo</div><div style="color:#666">105 partidos - hasta 30 seg...</div></div>';
  try {{
    const r=await fetch('/simulate_tournament',{{method:'POST'}});
    const d=await r.json();
    if(d.error){{out.innerHTML='<div class="error">'+d.error+'</div>';return;}}
    let html='';

    // Groups
    html+='<div class="card"><h2 style="color:#fbbf24;font-size:1.1rem;margin-bottom:12px;text-align:center">\U0001f3c6 Fase de Grupos</h2>';
    html+='<div style="font-size:.7rem;color:#888;text-align:center;margin-bottom:8px"><span style="display:inline-block;width:10px;height:10px;background:#22c55e;border-radius:2px;margin-right:4px;vertical-align:middle"></span> Clasificado <span style="display:inline-block;width:10px;height:10px;background:#eab308;border-radius:2px;margin-right:4px;margin-left:12px;vertical-align:middle"></span> Mejor 3ro <span style="opacity:.5;margin-left:12px">Eliminado</span></div>';
    html+='<div class="group-grid">';
    for(const g of d.groups) {{
      let rows='<tr><th class="pos">#</th><th class="name">Equipo</th><th class="pts">Pts</th><th class="dg">DG</th><th class="gf">GF</th></tr>';
      let i=0;
      for(const t of g.standings) {{
        const cls=i<2?'qualified-1':(i==2 && g.third_advances?'qualified-3':'eliminated');
        rows+='<tr class="'+cls+'"><td class="pos">'+(i+1)+'</td><td class="name">'+t.flag+' '+t.name+'</td><td class="pts">'+t.points+'</td><td class="dg">'+(t.dg>0?'+':'')+t.dg+'</td><td class="gf">'+t.gf+'</td></tr>';
        i++;
      }}
      html+='<div class="group-card"><h3>Grupo '+g.letter+'</h3><table class="group-table">'+rows+'</table></div>';
    }}
    html+='</div></div>';

    // Knockouts - bracket tree
    html+='<div class="card"><h2 style="color:#fbbf24;font-size:1.1rem;margin-bottom:12px;text-align:center">\U0001f3c6 Fase Eliminatoria</h2><div class="bracket-wrap">'+renderBracket(d.knockouts)+'</div></div>';

    // Winner
    html+='<div class="card tournament-winner"><div class="trophy">\U0001f3c6</div><div class="wname">'+d.winner_flag+' '+d.winner+'</div><div class="sub">Campeon del Mundial 2026</div></div>';

    out.innerHTML=html;
    refreshHistory();
  }}catch(e){{out.innerHTML='<div class="error">Error: '+e.message+'</div>';}}
  finally{{btn.disabled=false;btn.textContent='\U0001f3c6 Simular Torneo Completo';}}
}}

async function refreshHistory() {{
  try{{
    const r=await fetch('/history');const d=await r.json();
    let html='';
    for(const h of d){{
      if(h.type==='match') html+='<div class="hist-item"><span class="hist-teams">'+h.team_a+' vs '+h.team_b+'</span><span class="hist-pct">'+h.win1_pct.toFixed(0)+'% / '+h.draw_pct.toFixed(0)+'% / '+h.win2_pct.toFixed(0)+'%</span><span class="hist-date">'+h.date+'</span></div>';
      else if(h.type==='tournament') html+='<div class="hist-item"><span class="hist-teams">\U0001f3c6 Torneo: '+h.winner+'</span><span class="hist-date">'+h.date+'</span></div>';
    }}
    const card=document.getElementById('historyCard');
    if(card&&html) card.querySelector('.hist-list').innerHTML=html;
    else if(html){{
      const sec='<div class="card" id="historyCard"><h2 style="color:#fbbf24;margin-bottom:16px;font-size:1.1rem">\U0001f4cb Historial</h2><div class="hist-list">'+html+'</div></div>';
      document.querySelector('.container').insertAdjacentHTML('beforeend',sec);
    }}
  }}catch(e){{}}
}}
</script>
</body>
</html>"""


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
    elo_a, elo_b = float(teams_df.loc[team_a, 'elo']), float(teams_df.loc[team_b, 'elo'])
    results = simulate_match(xg_a, xg_b, elo_a, elo_b, team_a, team_b, n=simulations)

    total = sum(results.values())
    win1 = sum(v for (g1, g2), v in results.items() if g1 > g2)
    draw = sum(v for (g1, g2), v in results.items() if g1 == g2)
    win2 = sum(v for (g1, g2), v in results.items() if g2 > g1)
    sorted_scores = sorted(results.items(), key=lambda x: -x[1])
    top_scorelines = [{"score": f"{s[0]}-{s[1]}", "pct": round(v / total * 100, 2)} for (s, v) in sorted_scores[:10]]

    save_history({"type": "match", "date": datetime.now().strftime("%d/%m %H:%M"),
        "team_a": team_a, "team_b": team_b, "xg_a": round(xg_a, 2), "xg_b": round(xg_b, 2),
        "win1_pct": round(win1 / total * 100, 2), "draw_pct": round(draw / total * 100, 2),
        "win2_pct": round(win2 / total * 100, 2), "simulations": simulations})

    return {"team_a": team_a, "team_b": team_b, "xg_a": round(xg_a, 2), "xg_b": round(xg_b, 2),
        "simulations": simulations, "win1_pct": round(win1 / total * 100, 2),
        "draw_pct": round(draw / total * 100, 2), "win2_pct": round(win2 / total * 100, 2),
        "top_scorelines": top_scorelines}


@app.post("/simulate_tournament")
def simulate_tournament():
    try:
        t = Tournament(name="misterclaude", model=model)
        t.simulate_tournament()
        t.export_results()

        # Group standings
        best_third_names = {third.name for third in t.thirds}
        groups_data = []
        for i, group in enumerate(t.GROUPS):
            order = t.group_orders[i]
            standings = []
            for j, team in enumerate(order):
                standings.append({
                    "name": team.name, "flag": flag(team.name),
                    "points": team.points, "dg": team.dg, "gf": team.gf,
                })
            third_advances = order[2].name in best_third_names if len(order) > 2 else False
            groups_data.append({
                "letter": group.letter,
                "standings": standings,
                "third_advances": third_advances,
            })

        # Knockout bracket per round
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

        winner_name = t.winner.name if t.winner else "Desconocido"
        save_history({"type": "tournament", "date": datetime.now().strftime("%d/%m %H:%M"), "winner": winner_name})

        return {"winner": winner_name, "winner_flag": flag(winner_name),
                "groups": groups_data, "knockouts": knockouts_data}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/history")
def get_history():
    return load_history()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

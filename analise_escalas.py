import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from PIL import Image
import io
import base64
import os
import json
import requests
from datetime import datetime
import uuid

# --------------------------
# Config / constantes
# --------------------------
st.set_page_config(page_title="Radar de Escalas de Avaliação", layout="wide", initial_sidebar_state="expanded")

ASELC_BLUE = "#2A327A"
HGP_GREEN = "#006B61"
HGP_YELLOW = "#E5B900"
TEXT_PRIMARY = "#1F2937"
ACCENT_COLOR = "#3B82F6"

DATA_STORE = Path("data_store.csv")
HISTORY_STORE = Path("history_store.csv")  # registra publicações (gists ou uploads)
HISTORY_SNAP_DIR = Path("history_snapshots")
HISTORY_SNAP_DIR.mkdir(exist_ok=True)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # se presente habilita publicar como gist
GIST_API = "https://api.github.com/gists"

REFERENCIAS_TEXT = {
    "Curta": {"ref": 2.0, "desc": "Curta (1–3d): 1–2 avaliações — compatível com internações de curta duração."},
    "Média": {"ref": 4.5, "desc": "Média (4–10d): 3–5 avaliações — faixa intermediária, revisar protocolos se necessário."},
    "Longa": {"ref": 8.0, "desc": "Longa (>10d): 6–10+ avaliações — esperado para internações prolongadas."}
}

# --------------------------
# CSS / header
# --------------------------
st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(135deg, #F5F7FA 0%, #EEF2F6 100%); }}
.main-header {{ background: linear-gradient(135deg, {HGP_YELLOW} 0%, #FFE6A7 100%); padding: 1.8rem 1.2rem; border-radius: 12px; display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:2rem; }}
.main-title {{ color: {ASELC_BLUE}; font-size:2.0rem; font-weight:900; margin:0; }}
.main-sub {{ color: {ASELC_BLUE}; margin-top:0.4rem; }}
.success-box {{ background:#D1FAE5; border-left:4px solid #10B981; padding:10px; border-radius:8px; margin:8px 0; }}
.section-title {{ font-weight:700; margin-top:12px; margin-bottom:6px; color:{TEXT_PRIMARY}; }}
.info-box {{ background:#EFF6FF; border-left:4px solid #3B82F6; padding:10px; border-radius:8px; margin:8px 0; }}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Helpers (data, plotting, sharing)
# --------------------------
def load_local_logos():
    logos = {}
    if Path("logo_aselc.png").exists():
        try: logos["aselc"] = Image.open("logo_aselc.png")
        except: pass
    if Path("logo_hgp.png").exists():
        try: logos["hgp"] = Image.open("logo_hgp.png")
        except: pass
    return logos

def normalize_col_name(name: str) -> str:
    s = str(name).strip().lower()
    for a,b in {"ã":"a","á":"a","é":"e","í":"i","ó":"o","ú":"u","ç":"c"}.items():
        s = s.replace(a,b)
    return s.replace(" ", "_")

def prepare_dataframe(df):
    df = df.copy()
    col_map = {c: normalize_col_name(c) for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    mapping = {}
    for c in ["setor","sector","unidade","department"]:
        if c in df.columns: mapping[c] = "Setor"; break
    for c in ["tipo_de_escala","tipo_escala","escala","tipodeescala","tipo_internacao"]:
        if c in df.columns: mapping[c] = "Tipo_Escala"; break
    for c in ["quantidade_de_escalas","qtd_escalas","escalas","quantidadeescalas"]:
        if c in df.columns: mapping[c] = "Qtd_Escalas"; break
    for c in ["pacientes_internados","qtd_pacientes","pacientes"]:
        if c in df.columns: mapping[c] = "Qtd_Pacientes"; break
    for c in ["mes","mês","month","data","periodo"]:
        if c in df.columns: mapping[c] = "Mes"; break
    df.rename(columns=mapping, inplace=True)
    expected = ["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=expected)
    df = df[expected].copy()
    df["Setor"] = df["Setor"].astype(str).str.strip().str.title()
    df["Tipo_Escala"] = df["Tipo_Escala"].astype(str).str.strip().str.title()
    df["Mes"] = df["Mes"].astype(str).str.strip().str.title()
    df["Qtd_Escalas"] = pd.to_numeric(df["Qtd_Escalas"], errors="coerce").fillna(0)
    df["Qtd_Pacientes"] = pd.to_numeric(df["Qtd_Pacientes"], errors="coerce").fillna(0)
    df = df[(df["Qtd_Escalas"] > 0) & (df["Qtd_Pacientes"] > 0)]
    return df.reset_index(drop=True)

def compute_metrics(df, sector_adjust_map):
    df = df.copy()
    if df.empty: return df
    df["Escalas_por_Paciente"] = df.apply(lambda r: round(r["Qtd_Escalas"]/r["Qtd_Pacientes"],2) if r["Qtd_Pacientes"]>0 else 0.0, axis=1)
    def factor_for_setor(setor):
        s = setor.lower() if isinstance(setor, str) else ""
        for k,v in sector_adjust_map.items():
            if k in s: return v
        return 1.0
    df["Fator_Ajuste"] = df["Setor"].apply(factor_for_setor)
    df["Mediana_Ajustada"] = (df["Escalas_por_Paciente"] * df["Fator_Ajuste"]).round(2)
    return df

def aggregate_for_dashboard(df_subset):
    grp = df_subset.groupby("Tipo_Escala").agg(
        Qtd_Escalas=("Qtd_Escalas","sum"),
        Qtd_Pacientes=("Qtd_Pacientes","max"),
        Fator_Ajuste=("Fator_Ajuste","first")
    ).reset_index()
    grp["Escalas_por_Paciente"] = grp.apply(lambda r: round(r["Qtd_Escalas"]/r["Qtd_Pacientes"],2) if r["Qtd_Pacientes"]>0 else 0.0, axis=1)
    grp["Mediana_Ajustada"] = (grp["Escalas_por_Paciente"] * grp["Fator_Ajuste"]).round(2)
    return grp

def make_radar_figure(escalas, valores_real, referencia_vals):
    fig = go.Figure()
    theta = escalas + ([escalas[0]] if escalas else [])
    for k,v in referencia_vals.items():
        r = [v]*len(theta)
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=f"{k} (ref.)", opacity=0.18))
    if valores_real:
        r_real = valores_real + [valores_real[0]]
        fig.add_trace(go.Scatterpolar(r=r_real, theta=theta, fill="toself", name="Mediana Ajustada", line=dict(color=ASELC_BLUE, width=3)))
    max_ref = max(referencia_vals.values()) if referencia_vals else 1
    max_val = max(max(valores_real) if valores_real else 1, max_ref)
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, max_val+2])), height=560, margin=dict(t=20))
    return fig

def make_bar_mean_chart(escalas, medias, title):
    fig = go.Figure()
    colors = [HGP_YELLOW if i%2==0 else HGP_GREEN for i in range(len(escalas))]
    fig.add_trace(go.Bar(x=escalas, y=medias, marker_color=colors, text=[f"{v:.2f}" for v in medias], textposition="outside"))
    fig.update_layout(title=title, yaxis_title="Escalas por Paciente", height=420, margin=dict(t=80))
    return fig

# --------------------------
# Sharing / Gist helpers
# --------------------------
def make_standalone_html(df, plots, title="Relatório", meta=None, include_plotlyjs="cdn"):
    meta = meta or {}
    now = datetime.now().isoformat(timespec="seconds")
    head = f"<meta charset='utf-8'><title>{title}</title>"
    head += "<style>body{font-family:Arial,Helvetica,sans-serif;padding:18px;} h1{color:#2A327A;} .meta{color:#555;margin-bottom:12px;} table{border-collapse:collapse;width:100%;} table th, table td{border:1px solid #ddd;padding:6px;}</style>"
    header = f"<h1>{title}</h1><div class='meta'>Gerado: {now}"
    for k,v in meta.items(): header += f" | {k}: {v}"
    header += "</div>"
    table_html = df.to_html(index=False, classes='table', border=0)
    plots_html = ""
    for fig in plots:
        if fig is None: continue
        plots_html += pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs)
    html = f"<!doctype html><html><head>{head}</head><body>{header}{table_html}{plots_html}</body></html>"
    return html.encode("utf-8")

def create_gist_from_html(html_bytes, filename="relatorio.html", public=True, description="Relatório gerado"):
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN não definido")
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {"description": description, "public": public, "files": {filename: {"content": html_bytes.decode("utf-8")}}}
    resp = requests.post(GIST_API, headers=headers, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
    return resp.json()

def register_share_record(gist_id, raw_url, filename, public=True, source="gist"):
    now = datetime.now().isoformat(timespec="seconds")
    rec = {"share_id": gist_id, "raw_url": raw_url, "filename": filename, "timestamp": now, "public": public, "source": source}
    if HISTORY_STORE.exists():
        try:
            df = pd.read_csv(HISTORY_STORE)
            df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([rec])
    else:
        df = pd.DataFrame([rec])
    df.to_csv(HISTORY_STORE, index=False)

def load_share_index():
    if HISTORY_STORE.exists():
        try:
            return pd.read_csv(HISTORY_STORE).sort_values("timestamp", ascending=False).reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["share_id","raw_url","filename","timestamp","public","source"])
    return pd.DataFrame(columns=["share_id","raw_url","filename","timestamp","public","source"])

def delete_gist_by_id(gist_id):
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN não definido")
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    resp = requests.delete(f"{GIST_API}/{gist_id}", headers=headers, timeout=30)
    return resp.status_code == 204

# --------------------------
# Build UI header with logos
# --------------------------
logos = load_local_logos()
left_html = right_html = ""
if "aselc" in logos:
    buf = io.BytesIO(); logos["aselc"].save(buf, format="PNG"); left_html = base64.b64encode(buf.getvalue()).decode()
if "hgp" in logos:
    buf = io.BytesIO(); logos["hgp"].save(buf, format="PNG"); right_html = base64.b64encode(buf.getvalue()).decode()

st.markdown(f"""
<div class="main-header">
  <div style="width:160px;text-align:left">{f"<img src='data:image/png;base64,{left_html}' style='max-width:140px;height:auto;'/>" if left_html else ""}</div>
  <div style="text-align:center;">
    <div class="main-title">Painel de Escalas por Paciente</div>
    <div class="main-sub">Análise comparativa de avaliações — ASELC / HGP</div>
  </div>
  <div style="width:160px;text-align:right">{f"<img src='data:image/png;base64,{right_html}' style='max-width:140px;height:auto;'/>" if right_html else ""}</div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar (upload, history of shares, options)
# --------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configurações e Histórico")
    st.markdown("**Upload**")
    uploaded = st.file_uploader("Envie Excel ou CSV", type=["xlsx","xls","csv"], key="uploader")

    st.markdown("---")
    st.markdown("**Compartilhamentos**")
    shares_df = load_share_index()
    if not shares_df.empty:
        st.dataframe(shares_df[["timestamp","filename","raw_url","public","source"]], use_container_width=True)
        sel_share_row = st.selectbox("Selecionar compartilhamento para ações", options=shares_df.index.tolist(), format_func=lambda i: f"{shares_df.at[i,'timestamp']} — {shares_df.at[i,'filename']}")
        if sel_share_row is not None:
            rec = shares_df.loc[sel_share_row]
            st.write("Link:", rec.get("raw_url"))
            if GITHUB_TOKEN and st.button("🗑️ Apagar Gist selecionado"):
                gid = rec.get("share_id")
                try:
                    ok = delete_gist_by_id(gid)
                    if ok:
                        # remove from history file
                        new_df = shares_df[shares_df["share_id"] != gid]
                        new_df.to_csv(HISTORY_STORE, index=False)
                        st.success("Gist apagado e histórico atualizado.")
                        st.experimental_rerun()
                    else:
                        st.error("Falha ao apagar o gist (verifique token/permissões).")
                except Exception as e:
                    st.error(f"Erro ao apagar gist: {e}")
    else:
        st.info("Nenhum compartilhamento registrado ainda.")

    st.markdown("---")
    st.markdown("**Fatores de Ajuste**")
    fac_uti = st.number_input("UTI", value=1.20, step=0.01, format="%.2f")
    fac_emg = st.number_input("Emergência", value=1.15, step=0.01, format="%.2f")
    fac_enf = st.number_input("Enfermaria", value=1.00, step=0.01, format="%.2f")
    fac_amb = st.number_input("Ambulatório", value=1.00, step=0.01, format="%.2f")

# --------------------------
# Prepare session uploaded data
# --------------------------
if uploaded is not None:
    try:
        if str(uploaded.name).lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
    except Exception as e:
        raw = None
        st.sidebar.error(f"Erro ao ler arquivo: {e}")
    if raw is not None:
        df_uploaded = prepare_dataframe(raw)
        if df_uploaded.empty:
            st.sidebar.error("Nenhum dado válido após normalização.")
            st.session_state["df_uploaded_session"] = None
        else:
            st.session_state["df_uploaded_session"] = df_uploaded
            st.sidebar.success(f"{len(df_uploaded)} linhas preparadas na sessão (não salvas). Use 'Salvar histórico' para registrar snapshot se quiser.")
else:
    if "df_uploaded_session" not in st.session_state:
        st.session_state["df_uploaded_session"] = None

# --------------------------
# Build base_df (session upload preferred)
# --------------------------
sector_adjust_map = {"uti": fac_uti, "emerg": fac_emg, "enferm": fac_enf, "ambulatorio": fac_amb, "aloj": 0.9}

if st.session_state.get("df_uploaded_session") is not None and not st.session_state["df_uploaded_session"].empty:
    base_df = st.session_state["df_uploaded_session"].copy()
else:
    if DATA_STORE.exists():
        try:
            base_df = pd.read_csv(DATA_STORE)
            base_df = prepare_dataframe(base_df) if not base_df.empty else pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])
        except Exception:
            base_df = pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])
    else:
        base_df = pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])

if base_df is None or base_df.empty:
    st.markdown('<div style="padding:12px;background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:8px;">⚠️ Nenhum dado disponível para análise. Faça upload no menu lateral ou insira manualmente.</div>', unsafe_allow_html=True)
    st.stop()

df = compute_metrics(base_df, sector_adjust_map)

# --------------------------
# Dashboard filters
# --------------------------
st.markdown('<div class="section-title">📈 Dashboard Interativo</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2,2,1])
with col1:
    months = ["Todos"] + sorted(df["Mes"].dropna().unique().tolist()) if "Mes" in df.columns else ["Todos"]
    sel_month = st.selectbox("📅 Mês", months, key="sel_month")
with col2:
    sectors = sorted(df["Setor"].dropna().unique().tolist()) if "Setor" in df.columns else []
    sel_sector = st.selectbox("🏥 Setor", sectors, key="sel_sector") if sectors else st.text_input("🏥 Setor", value="")
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Gerar snapshot temporário (sessão)"):
        st.session_state["temp_snapshot_time"] = datetime.now().isoformat(timespec="seconds")
        st.success("Snapshot temporário gerado na sessão.")

if sel_month == "Todos":
    subset = df[df["Setor"] == sel_sector].copy() if sel_sector else df.copy()
    periodo_txt = "Todos os Meses"
else:
    subset = df[(df["Mes"] == sel_month) & (df["Setor"] == sel_sector)].copy()
    periodo_txt = sel_month

if subset.empty:
    st.markdown('<div class="info-box">ℹ️ Sem dados para os filtros selecionados.</div>', unsafe_allow_html=True)
    st.stop()

if "Tipo_Escala" not in subset.columns:
    st.error("Coluna 'Tipo_Escala' ausente. Verifique o arquivo de entrada.")
    st.stop()

group = aggregate_for_dashboard(subset)

# reactive visual card
visual = st.empty()
visual.markdown(f'<div class="success-box"><b>📊 Visualizando:</b> {periodo_txt} — {sel_sector}</div>', unsafe_allow_html=True)

# charts
escalas = group["Tipo_Escala"].tolist()
medianas = group["Mediana_Ajustada"].tolist()
pacientes = group["Qtd_Pacientes"].tolist()
medias_por_paciente = group["Escalas_por_Paciente"].tolist()

radar_fig = None
if len(escalas) >= 3:
    referencia_vals = {k: v["ref"] for k,v in REFERENCIAS_TEXT.items()}
    radar_fig = make_radar_figure(escalas, medianas, referencia_vals)
    st.plotly_chart(radar_fig, use_container_width=True)
else:
    st.markdown('<div class="info-box">ℹ️ Poucas escalas para radar. Exibindo complementos gráficos.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
bar_mean_fig = make_bar_mean_chart(escalas, medias_por_paciente, title=f"Média de Escalas por Paciente — {periodo_txt} / {sel_sector}")
st.plotly_chart(bar_mean_fig, use_container_width=True)

# --------------------------
# Share: generate HTML, download, publish gist
# --------------------------
st.markdown('<div class="section-title">📤 Compartilhar análise atual</div>', unsafe_allow_html=True)

export_df = df[["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes","Escalas_por_Paciente","Fator_Ajuste","Mediana_Ajustada"]].round(2)

meta = {"Mês": periodo_txt, "Setor": sel_sector}
html_bytes = make_standalone_html(df=export_df, plots=[radar_fig, bar_mean_fig], title=f"Relatório — {sel_sector} — {periodo_txt}", meta=meta, include_plotlyjs="cdn")

c1, c2 = st.columns([1,1])
with c1:
    st.download_button("⬇️ Baixar página HTML", data=html_bytes, file_name=f"relatorio_{sel_sector}_{periodo_txt}.html".replace(" ","_"), mime="text/html")
with c2:
    if GITHUB_TOKEN:
        if st.button("📤 Publicar como Gist (público)"):
            try:
                resp = create_gist_from_html(html_bytes, filename=f"relatorio_{sel_sector}_{periodo_txt}.html".replace(" ","_"), public=True, description=f"Relatório {sel_sector} {periodo_txt}")
                files = resp.get("files", {})
                first_file = list(files.keys())[0] if files else None
                raw_url = files[first_file]["raw_url"] if first_file else resp.get("html_url")
                gist_id = resp.get("id")
                register_share_record(gist_id, raw_url, first_file or f"relatorio_{uuid.uuid4().hex[:6]}.html", public=True, source="gist")
                st.success("Publicado como Gist com sucesso")
                st.write(raw_url)
            except Exception as e:
                st.error(f"Erro ao publicar Gist: {e}")
    else:
        st.info("Publicar como Gist exige GITHUB_TOKEN no ambiente (configure para habilitar).")

# --------------------------
# Metrics and details
# --------------------------
st.markdown('<div class="section-title">📊 Métricas Resumidas</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Total Escalas", f"{int(group['Qtd_Escalas'].sum())}")
rep_pat = int(group['Qtd_Pacientes'].max()) if not group.empty else 0
c2.metric("👥 Pacientes", f"{rep_pat}")
c3.metric("📈 Média Geral", f"{group['Escalas_por_Paciente'].mean():.2f}" if not group.empty else "0.00")
c4.metric("⚖️ Média Ajustada", f"{group['Mediana_Ajustada'].mean():.2f}" if not group.empty else "0.00")

with st.expander("📚 Dados Detalhados", expanded=False):
    st.latex(r"\text{Mediana Ajustada} = \left(\frac{\text{Qtd Escalas}}{\text{Qtd Pacientes}}\right) \times \text{Fator}")
    debug_df = group.copy()
    debug_df["Passo a Passo"] = debug_df.apply(lambda r: f"({int(r['Qtd_Escalas'])} ÷ {int(r['Qtd_Pacientes'])}) × {r['Fator_Ajuste']:.2f} = {r['Mediana_Ajustada']:.2f}", axis=1)
    st.dataframe(debug_df[["Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Escalas_por_Paciente","Fator_Ajuste","Mediana_Ajustada","Passo a Passo"]], use_container_width=True)

# --------------------------
# Temporal
# --------------------------
if "Mes" in df.columns and df["Mes"].nunique() > 1:
    st.markdown('<div class="section-title">📅 Análise Temporal</div>', unsafe_allow_html=True)
    temporal_data = df[df["Setor"] == sel_sector].groupby(["Mes","Tipo_Escala"]).agg(Qtd_Escalas=("Qtd_Escalas","sum"), Qtd_Pacientes=("Qtd_Pacientes","max")).reset_index()
    temporal_data["Escalas_por_Paciente"] = temporal_data.apply(lambda r: round(r["Qtd_Escalas"]/r["Qtd_Pacientes"],2) if r["Qtd_Pacientes"]>0 else 0.0, axis=1)
    fig_temporal = go.Figure()
    for escala in temporal_data["Tipo_Escala"].unique():
        d = temporal_data[temporal_data["Tipo_Escala"]==escala].sort_values("Mes")
        fig_temporal.add_trace(go.Scatter(x=d["Mes"], y=d["Escalas_por_Paciente"], mode="lines+markers", name=escala))
    fig_temporal.update_layout(title=f"Evolução Temporal — {sel_sector}", xaxis_title="Mês", yaxis_title="Escalas por Paciente", height=420)
    st.plotly_chart(fig_temporal, use_container_width=True)

st.markdown("---")
st.markdown(""
"<div style='text-align:center;color:#6B7280;padding:8px;'>" 
     <p style='margin:0;font-size:0.9rem;'>
         <b>Sistema de Análise de Escalas de Avaliação</b><br>
         ASELC / HGP — Desenvolvido para Gestão Hospitalar<br>
         © 2025 
    </p>
</div>
"", unsafe_allow_html=True)
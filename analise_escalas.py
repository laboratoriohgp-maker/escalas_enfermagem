import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import io
import requests, base64, datetime as st
from datetime import datetime
import uuid

# Optional PDF generation imports
try:
    import plotly.io as pio
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --------------------------
# Config / constantes
# --------------------------
st.set_page_config(page_title="Radar de Escalas de Avaliação", layout="wide", initial_sidebar_state="expanded")

ASELC_BLUE = "#2A327A"
HGP_GREEN = "#006B61"
HGP_YELLOW = "#E5B900"
TEXT_PRIMARY = "#1F2937"
ACCENT_COLOR = "#3B82F6"

DATA_STORE = Path("data_store.csv")            # legacy
HISTORY_STORE = Path("history_store.csv")     # index of snapshots
HISTORY_DIR = Path("history_snapshots")       # folder with snapshot files
HISTORY_DIR.mkdir(exist_ok=True)

REFERENCIAS_TEXT = {
    "Curta": {"ref": 2.0, "desc": "Curta (1–3d): 1–2 avaliações — compatível com internações de curta duração."},
    "Média": {"ref": 4.5, "desc": "Média (4–10d): 3–5 avaliações — faixa intermediária, revisar protocolos se necessário."},
    "Longa": {"ref": 8.0, "desc": "Longa (>10d): 6–10+ avaliações — esperado para internações prolongadas."}
}

# --------------------------
# Estilos
# --------------------------
st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(135deg, #F5F7FA 0%, #EEF2F6 100%); }}
.main-header {{ background: linear-gradient(135deg, {HGP_YELLOW} 0%, #FFE6A7 100%); padding: 2.0rem 1.2rem; border-radius: 12px; display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:2rem; }}
.main-title {{ color: {ASELC_BLUE}; font-size:2.2rem; font-weight:900; margin:0; }}
.main-sub {{ color: {ASELC_BLUE}; margin-top:0.4rem; }}
.success-box {{ background:#D1FAE5; border-left:4px solid #10B981; padding:10px; border-radius:8px; margin:8px 0; }}
.section-title {{ font-weight:700; margin-top:12px; margin-bottom:6px; color:{TEXT_PRIMARY}; }}
.info-box {{ background:#EFF6FF; border-left:4px solid #3B82F6; padding:10px; border-radius:8px; margin:8px 0; }}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Helpers
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

def safe_read_excel(uploaded_file):
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler Excel: {e}")
        return None

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
    for c in ["tipo_de_escala","tipo_escala","escala","tipodeescala"]:
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
        st.warning(f"Colunas faltando: {', '.join(missing)}. Retornando DataFrame padronizado vazio.")
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

# =============================
# 🔄 SINCRONIZAÇÃO COM GITHUB
# =============================

def upload_snapshot_to_github(df, filename=None):
    """Envia o DataFrame como CSV para o repositório GitHub configurado"""
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    if not filename:
        filename = f"snapshot_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"

    # Converte o dataframe em base64
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv_bytes).decode('utf-8')

    # Endpoint da API do GitHub
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {"Authorization": f"token {token}"}
    data = {
        "message": f"Novo snapshot {filename}",
        "content": b64
    }

    response = requests.put(url, headers=headers, json=data)
    if response.status_code in (200, 201):
        st.success(f"✅ Snapshot '{filename}' salvo no GitHub com sucesso!")
    else:
        st.error(f"❌ Falha ao salvar snapshot: {response.status_code}\n{response.text}")


def save_history_snapshot(df_snapshot, source_name="uploaded"):
    now = datetime.now().isoformat(timespec="seconds")
    snap_id = str(uuid.uuid4())[:8]
    meta = {"snapshot_id": snap_id, "timestamp": now, "source": source_name, "n_rows": len(df_snapshot)}
    hist_df = pd.DataFrame([meta])
    if HISTORY_STORE.exists():
        try:
            prev = pd.read_csv(HISTORY_STORE)
            hist_df = pd.concat([prev, hist_df], ignore_index=True)
        except Exception:
            pass
    hist_df.to_csv(HISTORY_STORE, index=False)
    fname = HISTORY_DIR / f"history_snapshot_{snap_id}.csv"
    df_snapshot.to_csv(fname, index=False)
    return snap_id, fname.name

def load_history_index():
    if HISTORY_STORE.exists():
        try:
            return pd.read_csv(HISTORY_STORE)
        except Exception:
            return pd.DataFrame(columns=["snapshot_id","timestamp","source","n_rows"])
    return pd.DataFrame(columns=["snapshot_id","timestamp","source","n_rows"])

def delete_history_snapshots(ids):
    idx = load_history_index()
    if idx.empty: return 0
    to_keep = idx[~idx["snapshot_id"].isin(ids)]
    to_delete = idx[idx["snapshot_id"].isin(ids)]
    # delete files
    deleted = 0
    for sid in to_delete["snapshot_id"].tolist():
        f = HISTORY_DIR / f"history_snapshot_{sid}.csv"
        try:
            if f.exists(): f.unlink(); deleted += 1
        except Exception:
            pass
    to_keep.to_csv(HISTORY_STORE, index=False)
    return deleted

def load_snapshot_df(snapshot_id):
    f = HISTORY_DIR / f"history_snapshot_{snapshot_id}.csv"
    if f.exists():
        try:
            df = pd.read_csv(f)
            df = prepare_dataframe(df) if not df.empty else df
            return df
        except Exception:
            return pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])
    return pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])

def df_to_download_bytes(df, fmt="csv"):
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), "text/csv"
    else:
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return towrite.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

def download_link_df(df, name="export.csv"):
    bts, mime = df_to_download_bytes(df, "csv")
    b64 = base64.b64encode(bts).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}" style="text-decoration:none;background:{ACCENT_COLOR};color:white;padding:6px 10px;border-radius:6px;font-weight:700;">⬇️ Baixar CSV</a>'
    return href

# Plot helpers
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
# UI Header
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
# Sidebar: upload, histórico com seleção e exclusão
# --------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configurações e Histórico")
    st.markdown("**Upload**")
    uploaded = st.file_uploader("Envie Excel ou CSV", type=["xlsx","xls","csv"], key="uploader")

    st.markdown("---")
    st.markdown("**Histórico salvo (snapshots)**")
    hist_index = load_history_index()
    if not hist_index.empty:
        hist_index_sorted = hist_index.sort_values("timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(hist_index_sorted, use_container_width=True)
        # multi-select for analysis
        sel_ids = st.multiselect("Selecione 1 ou mais snapshots para análise", options=hist_index_sorted["snapshot_id"].tolist(), default=[], help="Ao selecionar, apenas os snapshots escolhidos serão usados na análise quando 'Usar snapshots selecionados' for clicado.")
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("🗑️ Apagar snapshots selecionados"):
                if not sel_ids:
                    st.warning("Nenhum snapshot selecionado para apagar.")
                else:
                    deleted = delete_history_snapshots(sel_ids)
                    st.success(f"{deleted} snapshots apagados.")
                    st.experimental_rerun()
        with col2:
            if st.button("✅ Usar snapshots selecionados na análise"):
                if not sel_ids:
                    st.warning("Selecione ao menos um snapshot para usar.")
                else:
                    st.session_state["use_snapshots_ids"] = sel_ids
                    st.success(f"{len(sel_ids)} snapshots marcados para uso na análise atual.")
    else:
        st.info("Nenhum snapshot salvo ainda.")

    st.markdown("---")
    st.markdown("**Salvar histórico (snapshot)**")
    hist_source = st.text_input("Fonte (nome do arquivo ou descrição)", value="uploaded", help="Nome do arquivo ou descrição do snapshot")
    if st.button("💾 Salvar snapshot"):
        # df_atual é o dataframe com os dados que você quer salvar
        upload_snapshot_to_github(df)
    st.divider()
    st.subheader("📁 Snapshots disponíveis no GitHub")

    snapshots = listar_snapshots_github()
    if snapshots:
        for s in snapshots:
            st.write("📄", s)
    else:
        st.info("Nenhum snapshot disponível ainda.")

    st.markdown("---")
    st.markdown("**Opções**")
    include_history_all = st.checkbox("Incluir todo histórico salvo na análise (cuidado: pode duplicar dados)", value=False)
    st.markdown("Observação: uploads não salvam automaticamente; use 'Salvar histórico' para registrar um snapshot.")
    st.markdown("---")
    st.markdown("**Fatores de Ajuste**")
    fac_uti = st.number_input("UTI", value=1.20, step=0.01, format="%.2f")
    fac_emg = st.number_input("Emergência", value=1.15, step=0.01, format="%.2f")
    fac_enf = st.number_input("Enfermaria", value=1.00, step=0.01, format="%.2f")
    fac_amb = st.number_input("Ambulatório", value=1.00, step=0.01, format="%.2f")

# --------------------------
# Prepare session uploaded data (do not auto-save)
# --------------------------
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded) if str(uploaded.name).lower().endswith(".csv") else safe_read_excel(uploaded)
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
            st.sidebar.success(f"{len(df_uploaded)} linhas preparadas na sessão (não salvas). Para salvar clique em 'Salvar histórico'.")
else:
    if "df_uploaded_session" not in st.session_state:
        st.session_state["df_uploaded_session"] = None

# --------------------------
# Build analysis base_df according to user choices
# --------------------------
sector_adjust_map = {"uti": fac_uti, "emerg": fac_emg, "enferm": fac_enf, "ambulatorio": fac_amb, "aloj": 0.9}

# Start with session uploaded if exists, else fallback to data_store
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

# Option: include either all history, or only selected snapshots, or none
snap_ids_to_include = []
if "use_snapshots_ids" in st.session_state and st.session_state["use_snapshots_ids"]:
    snap_ids_to_include = st.session_state["use_snapshots_ids"]
elif include_history_all and not load_history_index().empty:
    snap_ids_to_include = load_history_index()["snapshot_id"].tolist()

# Load selected snapshots
snap_frames = []
for sid in snap_ids_to_include:
    sdf = load_snapshot_df(sid)
    if not sdf.empty:
        snap_frames.append(sdf)
if snap_frames:
    snap_concat = pd.concat(snap_frames, ignore_index=True)
    base_df = pd.concat([base_df, snap_concat], ignore_index=True)

# Safety: stop if nothing
if base_df is None or base_df.empty:
    st.markdown('<div style="padding:12px;background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:8px;">⚠️ Nenhum dado disponível para análise. Faça upload no menu lateral ou salve um snapshot.</div>', unsafe_allow_html=True)
    st.stop()

# Compute metrics (aplica fatores)
df = compute_metrics(base_df, sector_adjust_map)

# --------------------------
# Dashboard filters (reactive)
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
        st.success("Snapshot temporário criado na sessão.")

# Apply filters
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

# Aggregate for dashboard display (consistent)
group = aggregate_for_dashboard(subset)

# Reactive visual card
visual = st.empty()
visual.markdown(f'<div class="success-box"><b>📊 Visualizando:</b> {periodo_txt} — {sel_sector}</div>', unsafe_allow_html=True)

# Charts
escalas = group["Tipo_Escala"].tolist()
medianas = group["Mediana_Ajustada"].tolist()
pacientes = group["Qtd_Pacientes"].tolist()
medias_por_paciente = group["Escalas_por_Paciente"].tolist()

radar_fig = None
if len(escalas) >= 4:
    referencia_vals = {k: v["ref"] for k,v in REFERENCIAS_TEXT.items()}
    radar_fig = make_radar_figure(escalas, medianas, referencia_vals)
    st.plotly_chart(radar_fig, use_container_width=True)
else:
    st.markdown('<div class="info-box">ℹ️ Poucas escalas para radar. Exibindo complementos gráficos.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
bar_mean_fig = make_bar_mean_chart(escalas, medias_por_paciente, title=f"Média de Escalas por Paciente — {periodo_txt} / {sel_sector}")
st.plotly_chart(bar_mean_fig, use_container_width=True)

if st.checkbox("Mostrar número absoluto de pacientes por escala", value=False):
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=escalas, y=pacientes, text=[str(int(x)) for x in pacientes], textposition="outside"))
    fig_p.update_layout(title=f"Pacientes por Escala — {periodo_txt} / {sel_sector}", height=420)
    st.plotly_chart(fig_p, use_container_width=True)

# Metrics
st.markdown('<div class="section-title">📊 Métricas Resumidas</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Total Escalas", f"{int(group['Qtd_Escalas'].sum())}")
rep_pat = int(group['Qtd_Pacientes'].max()) if not group.empty else 0
c2.metric("👥 Pacientes (representativo)", f"{rep_pat}")
c3.metric("📈 Média Geral", f"{group['Escalas_por_Paciente'].mean():.2f}" if not group.empty else "0.00")
c4.metric("⚖️ Média Ajustada", f"{group['Mediana_Ajustada'].mean():.2f}" if not group.empty else "0.00")

# Data details (hidden)
with st.expander("📚 Dados Detalhados", expanded=False):
    st.latex(r"\text{Mediana Ajustada} = \left(\frac{\text{Qtd Escalas}}{\text{Qtd Pacientes}}\right) \times \text{Fator}")
    debug_df = group.copy()
    debug_df["Passo a Passo"] = debug_df.apply(lambda r: f"({int(r['Qtd_Escalas'])} ÷ {int(r['Qtd_Pacientes'])}) × {r['Fator_Ajuste']:.2f} = {r['Mediana_Ajustada']:.2f}", axis=1)
    st.dataframe(debug_df[["Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Escalas_por_Paciente","Fator_Ajuste","Mediana_Ajustada","Passo a Passo"]], use_container_width=True)

# References open
with st.expander("📖 Referências Metodológicas (aberto)", expanded=True):
    st.markdown(f"""
- **Curta (1–3d):** 1–2 avaliações — ref: {REFERENCIAS_TEXT['Curta']['ref']}  
  Contexto: internações rápidas.
- **Média (4–10d):** 3–5 avaliações — ref: {REFERENCIAS_TEXT['Média']['ref']}  
- **Longa (>10d):** 6–10+ avaliações — ref: {REFERENCIAS_TEXT['Longa']['ref']}  
""")

# Benchmarks interpretation
st.markdown('<div class="section-title">🚦 Interpretação Rápida (benchmarks)</div>', unsafe_allow_html=True)
for _, r in group.iterrows():
    val = r["Mediana_Ajustada"]
    if val <= REFERENCIAS_TEXT["Curta"]["ref"]:
        badge, phrase = "🟢", "Dentro do esperado (Curta)."
    elif val <= REFERENCIAS_TEXT["Média"]["ref"]:
        badge, phrase = "🟡", "Faixa média — avaliar protocolos."
    elif val <= REFERENCIAS_TEXT["Longa"]["ref"]:
        badge, phrase = "🟠", "Tendência para internação longa; investigar."
    else:
        badge, phrase = "🔴", "Acima da referência — investigação necessária."
    st.write(f"{badge} **{r['Tipo_Escala']}** — {val:.2f} — {phrase}")

# Export current analysis
st.markdown('<div class="section-title">📥 Exportar Dados da Análise</div>', unsafe_allow_html=True)
ex1, ex2 = st.columns(2)
with ex1:
    export_df = df[["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes","Escalas_por_Paciente","Fator_Ajuste","Mediana_Ajustada"]].round(2)
    st.markdown(download_link_df(export_df, name=f"escalas_analise_{sel_sector}_{periodo_txt}.csv".replace(' ','_')), unsafe_allow_html=True)
with ex2:
    st.info("Exportar histórico disponível no sidebar.")

# Temporal if available
if "Mes" in df.columns and df["Mes"].nunique() > 1:
    st.markdown('<div class="section-title">📅 Análise Temporal</div>', unsafe_allow_html=True)
    temp = df[df["Setor"] == sel_sector].groupby(["Mes","Tipo_Escala"]).agg(Qtd_Escalas=("Qtd_Escalas","sum"), Qtd_Pacientes=("Qtd_Pacientes","max")).reset_index()
    temp["Escalas_por_Paciente"] = temp.apply(lambda r: round(r["Qtd_Escalas"]/r["Qtd_Pacientes"],2) if r["Qtd_Pacientes"]>0 else 0.0, axis=1)
    figt = go.Figure()
    for escala in temp["Tipo_Escala"].unique():
        d = temp[temp["Tipo_Escala"]==escala].sort_values("Mes")
        figt.add_trace(go.Scatter(x=d["Mes"], y=d["Escalas_por_Paciente"], mode="lines+markers", name=escala))
    figt.update_layout(title=f"Evolução Temporal — {sel_sector}", xaxis_title="Mês", yaxis_title="Escalas por Paciente", height=420)
    st.plotly_chart(figt, use_container_width=True)

st.markdown("---")
st.markdown("""
"<div style='text-align:center;color:#6B7280;padding:8px;'>" 
     <p style='margin:0;font-size:0.9rem;'>
         <b>Sistema de Análise de Escalas de Avaliação</b><br>
         ASELC / HGP — Desenvolvido para Gestão Hospitalar<br>
         © 2025 
    </p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import io
import datetime
import base64
import os
import uuid
import requests

# ================================
# 🔒 CONFIGURAÇÃO GERAL DO APP
# ================================
st.set_page_config(page_title="Radar de Escalas de Avaliação", layout="wide", initial_sidebar_state="expanded")

ASELC_BLUE = "#2A327A"
HGP_GREEN = "#006B61"
HGP_YELLOW = "#E5B900"
TEXT_PRIMARY = "#1F2937"
ACCENT_COLOR = "#3B82F6"

DATA_STORE = Path("data_store.csv")
HISTORY_STORE = Path("history_store.csv")
HISTORY_DIR = Path("history_snapshots")
HISTORY_DIR.mkdir(exist_ok=True)

REFERENCIAS_TEXT = {
    "Curta": {"ref": 2.0, "desc": "Curta (1–3d): 1–2 avaliações — compatível com internações curtas."},
    "Média": {"ref": 4.5, "desc": "Média (4–10d): 3–5 avaliações — faixa intermediária."},
    "Longa": {"ref": 8.0, "desc": "Longa (>10d): 6–10+ avaliações — esperado para internações prolongadas."}
}

# ================================
# 🎨 ESTILOS
# ================================
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

# ================================
# 🔧 FUNÇÕES DE SUPORTE
# ================================
def load_local_logos():
    logos = {}
    for name in ["logo_aselc.png", "logo_hgp.png"]:
        if Path(name).exists():
            try:
                logos[name.split("_")[1].split(".")[0]] = Image.open(name)
            except:
                pass
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
    for c in ["tipo_de_escala","tipo_escala","escala"]:
        if c in df.columns: mapping[c] = "Tipo_Escala"; break
    for c in ["quantidade_de_escalas","qtd_escalas","escalas"]:
        if c in df.columns: mapping[c] = "Qtd_Escalas"; break
    for c in ["pacientes_internados","qtd_pacientes","pacientes"]:
        if c in df.columns: mapping[c] = "Qtd_Pacientes"; break
    for c in ["mes","mês","month","periodo"]:
        if c in df.columns: mapping[c] = "Mes"; break
    df.rename(columns=mapping, inplace=True)

    expected = ["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"]
    for col in expected:
        if col not in df.columns:
            df[col] = None

    df = df[expected].copy()
    df["Qtd_Escalas"] = pd.to_numeric(df["Qtd_Escalas"], errors="coerce").fillna(0)
    df["Qtd_Pacientes"] = pd.to_numeric(df["Qtd_Pacientes"], errors="coerce").fillna(0)
    df = df[(df["Qtd_Escalas"] > 0) & (df["Qtd_Pacientes"] > 0)]
    return df.reset_index(drop=True)

def compute_metrics(df, sector_adjust_map):
    if df.empty: return df
    df = df.copy()
    df["Escalas_por_Paciente"] = df["Qtd_Escalas"] / df["Qtd_Pacientes"]
    def factor_for_setor(setor):
        s = setor.lower() if isinstance(setor, str) else ""
        for k,v in sector_adjust_map.items():
            if k in s: return v
        return 1.0
    df["Fator_Ajuste"] = df["Setor"].apply(factor_for_setor)
    df["Mediana_Ajustada"] = (df["Escalas_por_Paciente"] * df["Fator_Ajuste"]).round(2)
    return df

# ================================
# ☁️ FUNÇÕES DE SINCRONIZAÇÃO GITHUB
# ================================
def upload_snapshot_to_github(df):
    """Envia o DataFrame como CSV para o repositório GitHub configurado via st.secrets"""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except KeyError:
        st.error("⚠️ Configure GITHUB_TOKEN e GITHUB_REPO em st.secrets antes de usar esta função.")
        return

    filename = f"snapshot_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode("utf-8")

    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {"Authorization": f"token {token}"}
    data = {"message": f"Novo snapshot {filename}", "content": b64}

    r = requests.put(url, headers=headers, json=data)
    if r.status_code in (200, 201):
        st.success(f"✅ Snapshot '{filename}' salvo com sucesso no GitHub!")
    else:
        st.error(f"❌ Falha ao salvar snapshot ({r.status_code}): {r.text}")

def listar_snapshots_github():
    """Lista arquivos CSV do repositório"""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except KeyError:
        return []

    url = f"https://api.github.com/repos/{repo}/contents"
    headers = {"Authorization": f"token {token}"}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return [f["name"] for f in r.json() if f["name"].endswith(".csv")]
        else:
            return []
    except Exception:
        return []

# ================================
# 🧩 INTERFACE PRINCIPAL
# ================================
logos = load_local_logos()

st.markdown(f"""
<div class="main-header">
  <div style="width:160px;">{f"<img src='data:image/png;base64,{base64.b64encode(open('logo_aselc.png','rb').read()).decode()}' style='max-width:140px;'/>" if Path('logo_aselc.png').exists() else ""}</div>
  <div style="text-align:center;">
    <div class="main-title">Painel de Escalas por Paciente</div>
    <div class="main-sub">Análise comparativa de avaliações — ASELC / HGP</div>
  </div>
  <div style="width:160px;text-align:right;">{f"<img src='data:image/png;base64,{base64.b64encode(open('logo_hgp.png','rb').read()).decode()}' style='max-width:140px;'/>" if Path('logo_hgp.png').exists() else ""}</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    uploaded = st.file_uploader("📤 Envie arquivo Excel ou CSV", type=["xlsx","xls","csv"])

    st.markdown("---")
    st.subheader("💾 Snapshots (GitHub)")
    if st.button("Salvar snapshot no GitHub"):
        if "df" in locals() and not df.empty:
            upload_snapshot_to_github(df)
        else:
            st.warning("Nenhum dado processado disponível para salvar.")
    st.markdown("---")
    st.subheader("📂 Snapshots disponíveis")
    snaps = listar_snapshots_github()
    if snaps:
        for s in snaps:
            st.write("📄", s)
    else:
        st.info("Nenhum snapshot encontrado.")

# ================================
# 📊 ANÁLISE DE DADOS
# ================================
if uploaded is not None:
    raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else safe_read_excel(uploaded)
    if raw is not None:
        df = prepare_dataframe(raw)
        if not df.empty:
            st.success("✅ Dados carregados com sucesso!")
        else:
            st.error("Nenhum dado válido encontrado no arquivo enviado.")
else:
    st.stop()

sector_adjust_map = {"uti":1.2, "emerg":1.15, "enferm":1.0, "ambulatorio":1.0}
df = compute_metrics(df, sector_adjust_map)

st.markdown("### 📈 Resultados")
st.dataframe(df.head(), use_container_width=True)

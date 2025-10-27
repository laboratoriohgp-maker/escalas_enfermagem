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

# =============================
# 🔄 SINCRONIZAÇÃO COM GITHUB
# =============================

def upload_snapshot_to_github(df, filename=None):
    """Envia o DataFrame como CSV para o repositório GitHub configurado"""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except KeyError:
        st.error("⚠️ Configure 'GITHUB_TOKEN' e 'GITHUB_REPO' em st.secrets para usar esta função.")
        return

    if not filename:
        filename = f"snapshot_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv_bytes).decode('utf-8')

    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {"Authorization": f"token {token}"}
    data = {"message": f"Novo snapshot {filename}", "content": b64}

    try:
        response = requests.put(url, headers=headers, json=data)
        if response.status_code in (200, 201):
            st.success(f"✅ Snapshot '{filename}' salvo no GitHub com sucesso!")
        else:
            st.error(f"❌ Falha ao salvar snapshot: {response.status_code} — {response.text}")
    except Exception as e:
        st.error(f"Erro ao enviar snapshot: {e}")

def listar_snapshots_github():
    """Lista arquivos CSV salvos no repositório GitHub configurado"""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except KeyError:
        st.error("⚠️ Configure 'GITHUB_TOKEN' e 'GITHUB_REPO' em st.secrets para listar snapshots.")
        return []

    url = f"https://api.github.com/repos/{repo}/contents"
    headers = {"Authorization": f"token {token}"}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return [f["name"] for f in r.json() if f["name"].endswith(".csv")]
        else:
            st.error(f"Erro ao buscar snapshots no GitHub: {r.status_code}")
            return []
    except Exception as e:
        st.error(f"Erro ao listar snapshots: {e}")
        return []

def load_snapshot_from_github(snapshot_name):
    """Carrega CSV do GitHub e retorna DataFrame"""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except KeyError:
        st.error("⚠️ Configure 'GITHUB_TOKEN' e 'GITHUB_REPO' em st.secrets")
        return pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])
    
    url = f"https://api.github.com/repos/{repo}/contents/{snapshot_name}"
    headers = {"Authorization": f"token {token}"}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            content = r.json()["content"]
            decoded = base64.b64decode(content)
            df = pd.read_csv(io.BytesIO(decoded))
            return prepare_dataframe(df)
        else:
            st.warning(f"Falha ao carregar {snapshot_name} do GitHub: {r.status_code}")
            return pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])
    except Exception as e:
        st.error(f"Erro ao baixar snapshot do GitHub: {e}")
        return pd.DataFrame(columns=["Setor","Tipo_Escala","Qtd_Escalas","Qtd_Pacientes","Mes"])

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
HISTORY_STORE = Path("history_store.csv")     
HISTORY_DIR = Path("history_snapshots")       
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
# 🚀 INÍCIO DO STREAMLIT APP
# =============================
st.title("📊 Radar de Escalas de Avaliação")
uploaded_file = st.file_uploader("Carregar arquivo Excel com escalas", type=["xlsx","xls"])
sector_adjust_map = {"clinica":1.0,"uti":1.5,"emergencia":1.2}

if uploaded_file:
    df_raw = safe_read_excel(uploaded_file)
    if df_raw is not None:
        df = prepare_dataframe(df_raw)
        df_metrics = compute_metrics(df, sector_adjust_map)
        st.dataframe(df_metrics)

        st.subheader("Resumo por Tipo de Escala")
        df_dashboard = aggregate_for_dashboard(df_metrics)
        st.dataframe(df_dashboard)

        if st.button("💾 Salvar Snapshot no GitHub"):
            upload_snapshot_to_github(df_metrics)

# Listar snapshots existentes
st.sidebar.header("Snapshots Existentes no GitHub")
snapshots = listar_snapshots_github()
selected_snapshot = st.sidebar.selectbox("Carregar snapshot existente", [""] + snapshots)
if selected_snapshot:
    df_loaded = load_snapshot_from_github(selected_snapshot)
    st.sidebar.write("✅ Snapshot carregado:")
    st.sidebar.dataframe(df_loaded)

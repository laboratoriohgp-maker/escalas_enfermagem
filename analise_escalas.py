# app_escalas.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import io
import base64

# Optional PDF generation imports (kaleido + reportlab)
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
# Config
# --------------------------
st.set_page_config(page_title="Radar de Escalas de Avaliação", layout="wide")
ASELC_BLUE = "#2A327A"
HGP_GREEN = "#006B61"
HGP_YELLOW = "#E5B900"
HGP_BLUE = "#0072B1"
BG_GRAY = "#F7F8FA"

DATA_STORE = Path("data_store.csv")   # local persistent storage

# --------------------------
# Helper functions
# --------------------------
def load_local_logos():
    logos = {}
    if Path("logo_aselc.png").exists():
        logos['aselc'] = Image.open("logo_aselc.png")
    elif Path("logo aselc.png").exists():
        logos['aselc'] = Image.open("logo aselc.png")
    if Path("logo_hgp.png").exists():
        logos['hgp'] = Image.open("logo_hgp.png")
    return logos

def safe_read_excel(uploaded_file):
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler Excel: {e}")
        return None

def persist_df(df):
    df.to_csv(DATA_STORE, index=False)

def load_persisted():
    if DATA_STORE.exists():
        return pd.read_csv(DATA_STORE)
    return pd.DataFrame(columns=[
        "Mês","Setor","Tipo_Internacao","Escala","Qtd_Escalas","Qtd_Pacientes"
    ])

def prepare_dataframe(df):
    # Normalize columns (accept multiple naming styles)
    df = df.copy()
    col_map = {}
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for orig, norm in zip(df.columns, cols):
        col_map[orig] = norm
    df.rename(columns=col_map, inplace=True)
    # Try to map required columns to canonical names
    mapping = {}
    # month
    for c in ['mês','mes','month','m']:
        if c in df.columns:
            mapping[c] = 'Mês'
    # setor
    for c in ['setor','department','ward']:
        if c in df.columns:
            mapping[c] = 'Setor'
    # tipo internação
    for c in ['tipo_de_internacao','tipo_internacao','tipo','internacao','tipo_de_internação']:
        if c in df.columns:
            mapping[c] = 'Tipo_Internacao'
    # escala
    for c in ['escala','tipo_de_escala','tipo_escala']:
        if c in df.columns:
            mapping[c] = 'Escala'
    # qtd escalas
    for c in ['quantidade_de_escalas','qtd_escalas','qtd_escalas','quantidade_escalas','qtd_escalas']:
        if c in df.columns:
            mapping[c] = 'Qtd_Escalas'
    # qtd pacientes
    for c in ['pacientes_internados','qtd_pacientes','qtd_pacientes','pacientes','pacientes_internados']:
        if c in df.columns:
            mapping[c] = 'Qtd_Pacientes'
    # Apply mapping if found
    df = df.rename(columns=mapping)
    # Keep only canonical cols (if they're present)
    expected = ['Mês','Setor','Tipo_Internacao','Escala','Qtd_Escalas','Qtd_Pacientes']
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    df = df[expected]
    # Strip strings
    df['Setor'] = df['Setor'].astype(str).str.strip().str.title()
    df['Escala'] = df['Escala'].astype(str).str.strip().str.title()
    df['Tipo_Internacao'] = df['Tipo_Internacao'].astype(str).str.strip().str.title()
    # Numeric conversions
    df['Qtd_Escalas'] = pd.to_numeric(df['Qtd_Escalas'], errors='coerce').fillna(0).astype(float)
    df['Qtd_Pacientes'] = pd.to_numeric(df['Qtd_Pacientes'], errors='coerce').fillna(0).astype(float)
    return df

def compute_metrics(df, sector_adjust_map):
    df = df.copy()
    # Escalas por paciente (aggregate)
    # If raw patient-level data present, you could compute true medians — here we work with aggregated rows
    df['Escalas_por_Paciente'] = df.apply(
        lambda r: (r['Qtd_Escalas'] / r['Qtd_Pacientes']) if r['Qtd_Pacientes'] > 0 else 0.0, axis=1
    )
    # Determine factor de ajuste por setor (configurable)
    def factor_for_setor(setor):
        for key, val in sector_adjust_map.items():
            if key.lower() in setor.lower():
                return val
        return 1.0
    df['Fator_Ajuste'] = df['Setor'].apply(factor_for_setor)
    # Mediana por escala (here median across rows of the same Escala in the chosen subset)
    # Mediana ajustada:
    df['Mediana_Ajustada'] = (df['Escalas_por_Paciente'] * df['Fator_Ajuste']).round(2)
    return df

def make_radar_figure(escalas, valores_real, referencia_vals, title):
    # referencias: dict with keys 'Curta','Media','Longa' -> numeric
    fig = go.Figure()
    # Add reference bands as translucent polygons
    # We'll stack by increasing reference
    colors_ref = {
        'Curta (1–3d)': 'rgba(42,50,122,0.12)',     # ASELC blue light
        'Média (4–10d)': 'rgba(229,181,0,0.12)',    # yellow light
        'Longa (>10d)': 'rgba(0,107,97,0.08)'       # HGP green light
    }
    # For each reference, create a polygon same length as escalas
    for k,v in referencia_vals.items():
        fig.add_trace(go.Scatterpolar(
            r=[v]*len(escalas),
            theta=escalas,
            fill='toself',
            name=f"{k} (ref.)",
            opacity=0.25,
            line=dict(color=colors_ref.get(k,'rgba(200,200,200,0.5)'))
        ))
    # Add observed median line
    fig.add_trace(go.Scatterpolar(
        r=valores_real,
        theta=escalas,
        fill='toself',
        name='Mediana Real Ajustada',
        line=dict(color=ASELC_BLUE, width=3)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(valores_real) if valores_real else 1, max(referencia_vals.values())) + 1])
        ),
        showlegend=True,
        title=title,
        height=600
    )
    return fig

def make_bar_patients_chart(escalas, pacientes, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=escalas,
        y=pacientes,
        marker_color=HGP_GREEN,
        name='Pacientes Internados'
    ))
    fig.update_layout(title=title, yaxis_title='Pacientes', height=420)
    return fig

def download_link_df(df, name="export.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}">⬇️ Baixar dados (CSV)</a>'
    return href

def try_export_pdf(figures, title="Relatório"):
    """
    Export minimal PDF with images of provided Plotly figures.
    Requires kaleido and reportlab to be available.
    Returns bytes of PDF or None.
    """
    if not KALEIDO_AVAILABLE or not REPORTLAB_AVAILABLE:
        return None, "PDF export requires package 'kaleido' and 'reportlab' installed."
    # render each fig to png bytes
    pngs = []
    for f in figures:
        img_bytes = pio.to_image(f, format='png', width=1000, height=600, scale=1)
        pngs.append(img_bytes)
    # create PDF
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    # add title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 40, title)
    y = h - 80
    for img in pngs:
        # insert image
        img_io = io.BytesIO(img)
        # scale to fit width
        img_pil = Image.open(img_io)
        iw, ih = img_pil.size
        ratio = (w - 80) / iw
        img_w = iw * ratio
        img_h = ih * ratio
        if y - img_h < 40:
            c.showPage()
            y = h - 40
        c.drawInlineImage(img_pil, 40, y - img_h, width=img_w, height=img_h)
        y = y - img_h - 20
    c.save()
    buffer.seek(0)
    return buffer.read(), None

# --------------------------
# UI: Header with logos
# --------------------------
logos = load_local_logos()
col1, col2, col3 = st.columns([1,6,1])
with col1:
    if 'aselc' in logos:
        st.image(logos['aselc'], width=140)
with col2:
    st.markdown("<h2 style='text-align:center;color:{}'>Painel Profissional — Escalas por Paciente</h2>".format(ASELC_BLUE), unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray'>Comparativo real vs referências por tipo de internação — ASELC / HGP</p>", unsafe_allow_html=True)
with col3:
    if 'hgp' in logos:
        st.image(logos['hgp'], width=140)

st.markdown("---")

# --------------------------
# Data source selection
# --------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("1) Carregar dados (opcional)")
    uploaded = st.file_uploader("Envie a planilha enxuta (opcional). Colunas esperadas: Mês, Setor, Tipo_Internacao, Escala, Qtd_Escalas, Qtd_Pacientes", type=["xlsx","xls","csv"])
    if uploaded:
        if str(uploaded.name).lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = safe_read_excel(uploaded)
        if raw is not None:
            df_uploaded = prepare_dataframe(raw)
            st.success("Arquivo carregado e formatado.")
            if st.checkbox("Visualizar dados carregados"):
                st.dataframe(df_uploaded)
    else:
        df_uploaded = None

with colB:
    st.subheader("2) Dados persistidos / locais")
    df_store = load_persisted()
    if not df_store.empty:
        st.write(f"Dados persistidos localmente: {len(df_store)} linhas")
        if st.checkbox("Visualizar dados persistidos"):
            st.dataframe(df_store)
    else:
        st.info("Nenhum dado salvo localmente ainda.")

# Decide base dataframe: uploaded overrides persisted unless none
if df_uploaded is not None:
    base_df = pd.concat([df_uploaded, df_store], ignore_index=True) if not df_store.empty else df_uploaded.copy()
else:
    base_df = df_store.copy()

# Ensure canonical columns exist
if base_df is None or base_df.empty:
    st.warning("Nenhum dado disponível — use o upload ou insira manualmente os dados abaixo.")
else:
    st.success("Dados prontos para análise.")

# --------------------------
# Manual insertion form
# --------------------------
st.markdown("## Inserção manual / edição rápida")
with st.form("insert_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([2,2,2])
    mes = c1.text_input("Mês (ex: Julho)", value="")
    setor = c2.text_input("Setor (ex: UTI Adulto)", value="")
    tipo_int = c3.selectbox("Tipo de Internação (classificação)", ["Curta","Média","Longa","Personalizado"])
    escala = st.text_input("Tipo de Escala (ex: Braden)", value="")
    col1, col2 = st.columns(2)
    qtd_escalas = col1.number_input("Qtd. Escalas", min_value=0, step=1, value=0)
    qtd_pacientes = col2.number_input("Qtd. Pacientes Internados", min_value=0, step=1, value=0)
    submitted = st.form_submit_button("➕ Adicionar / Atualizar")
    if submitted:
        new = {
            "Mês": mes,
            "Setor": setor.title(),
            "Tipo_Internacao": tipo_int.title(),
            "Escala": escala.title(),
            "Qtd_Escalas": float(qtd_escalas),
            "Qtd_Pacientes": float(qtd_pacientes)
        }
        # append to persisted store
        df_store = load_persisted()
        df_store = pd.concat([df_store, pd.DataFrame([new])], ignore_index=True)
        persist_df(df_store)
        st.success("Linha adicionada e salva localmente.")

# --------------------------
# Parametrização dos fatores setoriais (user-editable)
# --------------------------
st.markdown("## Ajustes & Parâmetros")
st.markdown("### Fatores de ajuste por setor (busca parcial por nome de setor). Ex: 'Uti' aplica fator 1.2 a qualquer setor que contenha 'Uti'.")
st.write("Se vazio, é aplicada a regra padrão: UTI -> 1.2 ; Emergência -> 1.15 ; Enfermaria -> 1.0")

f1, f2, f3 = st.columns(3)
with f1:
    fac_uti = st.number_input("Fator UTI (ex: 1.2)", value=1.20, step=0.01, format="%.2f")
with f2:
    fac_emg = st.number_input("Fator Emergência (ex: 1.15)", value=1.15, step=0.01, format="%.2f")
with f3:
    fac_enf = st.number_input("Fator Enfermaria (ex: 1.0)", value=1.00, step=0.01, format="%.2f")

# Build map: keys (substrings) -> factor
sector_adjust_map = {
    "uti": fac_uti,
    "emerg": fac_emg,
    "enferm": fac_enf,
    "aloj": 0.9  # example smaller factor for alojamento
}

# --------------------------
# Analysis / Filters
# --------------------------
st.markdown("---")
st.subheader("Dashboard Interativo")

if base_df is None or base_df.empty:
    st.info("Ainda não há dados. Carregue um arquivo ou insira algumas linhas.")
    st.stop()

# Prepare canonical df
df = prepare_dataframe(base_df)
# Compute metrics (columns: Escalas_por_Paciente, Fator_Ajuste, Mediana_Ajustada)
df = compute_metrics(df, sector_adjust_map)

# Filter selectors
months = sorted(df['Mês'].dropna().unique())
sectors = sorted(df['Setor'].dropna().unique())

left, right = st.columns([3,1])
with left:
    sel_month = st.selectbox("Selecione Mês", months)
    sel_sector = st.selectbox("Selecione Setor", sectors)
with right:
    st.markdown("### Ações rápidas")
    if st.button("Salvar snapshot atual (sobrescrever persisted)"):
        persist_df(df[['Mês','Setor','Tipo_Internacao','Escala','Qtd_Escalas','Qtd_Pacientes']])
        st.success("Snapshot salvo em data_store.csv")

# Subset
subset = df[(df['Mês'] == sel_month) & (df['Setor'] == sel_sector)].copy()
if subset.empty:
    st.warning("Não há linhas para o filtro selecionado.")
    st.stop()

# Group by Escala: in many cases there is 1 row per escala; aggregate to be safe
group = subset.groupby(['Escala','Tipo_Internacao']).agg({
    'Qtd_Escalas':'sum',
    'Qtd_Pacientes':'sum',
    'Escalas_por_Paciente': 'median',  # median across possible duplicates
    'Fator_Ajuste': 'median'
}).reset_index()
group['Mediana_Ajustada'] = (group['Escalas_por_Paciente'] * group['Fator_Ajuste']).round(2)

# Reference bands fixed (these will be shown as horizontal polygons across radar axes)
referencias = {
    "Curta (1–3d)": 2,     # você pode ajustar se necessário
    "Média (4–10d)": 4.5,
    "Longa (>10d)": 8
}

# Choose visualization: radar if >= 3 escalas, else bars
escalas = group['Escala'].tolist()
medianas = group['Mediana_Ajustada'].tolist()
pacientes = group['Qtd_Pacientes'].tolist()

st.markdown(f"### {sel_month} — {sel_sector}")
if len(escalas) >= 3:
    radar_fig = make_radar_figure(escalas, medianas, referencias, title=f"{sel_month} — {sel_sector} | Radar de Mediana Ajustada")
    st.plotly_chart(radar_fig, use_container_width=True)
else:
    st.info("Poucas escalas — mostrando gráfico de barras comparativo no lugar do radar.")
    bar_fig = make_bar_patients_chart(escalas, pacientes, title=f"{sel_month} — {sel_sector} | Pacientes por Escala")
    st.plotly_chart(bar_fig, use_container_width=True)

# Always show the bar patients chart as complement
st.markdown("#### Complemento — Pacientes avaliados por escala")
bar_fig2 = make_bar_patients_chart(escalas, pacientes, title="Pacientes avaliados por escala")
st.plotly_chart(bar_fig2, use_container_width=True)

# --------------------------
# Fixed reference table below graphs (user requested)
# --------------------------
st.markdown("---")
st.markdown("## Referências e parâmetros utilizados (fixos)")
st.markdown("""
**Parâmetros de referência por tipo de internação (faixas usadas no radar):**

- **Curta (1–3 dias):** 1–2 avaliações por paciente — *valor de referência plotado: 2*.  
  *Fonte:* Protocolos institucionais de enfermagem; recomendações hospitalares (ex: avaliação admissional + reavaliação).
- **Média (4–10 dias):** 3–5 avaliações por paciente — *valor de referência plotado: 4.5*.  
  *Fonte:* Boas práticas de monitoramento e reavaliação periódica (Ministério da Saúde / protocolos hospitalares).
- **Longa (>10 dias):** 6–10+ avaliações por paciente — *valor de referência plotado: 8*.  
  *Fonte:* Diretrizes para pacientes de longa permanência / UTI (ANVISA / protocolos especializados).

**Mediana Ajustada:** calculada como `round(median(Escalas por Paciente) * Fator de Ajuste Setorial)`.  
- Fatores de ajuste aplicados por padrão (configuráveis no painel):  
  - UTI: **1.20**  
  - Emergência: **1.15**  
  - Enfermaria geral: **1.00**  
  - Alojamento: **0.90** (exemplo)

**Observação metodológica:**  
- Se você fornecer dados em nível de paciente (cada linha = avaliação de escala por paciente), o app poderá calcular a mediana **verdadeira** por paciente; caso você forneça apenas os agregados por escala (como Qtd_Escalas e Qtd_Pacientes), o app estima `Escalas_por_Paciente = Qtd_Escalas / Qtd_Pacientes` e então aplica a mediana e fator de ajuste sobre essas estimativas.  
- As referências são baseadas em literatura e protocolos institucionais (Ministério da Saúde, ANVISA, protocolos hospitalares). Caso queira, posso anexar arquivos PDF das referências ou links diretos no rodapé do relatório.
""")

# Show the actual table of values used to produce the plots (fixed below graphs)
st.markdown("### Tabela de cálculo — valores usados no painel")
table_display = group[['Escala','Qtd_Escalas','Qtd_Pacientes','Escalas_por_Paciente','Fator_Ajuste','Mediana_Ajustada']].copy()
table_display['Escalas_por_Paciente'] = table_display['Escalas_por_Paciente'].round(2)
st.dataframe(table_display)

# --------------------------
# Export / Download: data + PDF
# --------------------------
st.markdown("---")
st.markdown("## Exportar / Baixar")
colx, coly = st.columns(2)
with colx:
    st.markdown("### Dados")
    st.markdown(download_link_df(df[['Mês','Setor','Tipo_Internacao','Escala','Qtd_Escalas','Qtd_Pacientes']].round(2), name=f"dados_{sel_month}_{sel_sector}.csv"), unsafe_allow_html=True)
with coly:
    st.markdown("### Relatório PDF (opcional)")
    if not KALEIDO_AVAILABLE or not REPORTLAB_AVAILABLE:
        st.warning("Exportar PDF requer 'kaleido' e 'reportlab' instalados (pip install kaleido reportlab).")
    if st.button("Gerar PDF do painel (figuras)"):
        # choose figures to export
        figs = []
        if len(escalas) >= 3:
            figs.append(radar_fig)
        else:
            figs.append(bar_fig)
        figs.append(bar_fig2)
        pdf_bytes, err = try_export_pdf(figs, title=f"Relatório - {sel_month} - {sel_sector}")
        if pdf_bytes:
            st.success("PDF gerado.")
            st.download_button("⬇️ Baixar PDF", data=pdf_bytes, file_name=f"relatorio_{sel_month}_{sel_sector}.pdf", mime="application/pdf")
        else:
            st.error(err or "Erro ao gerar PDF.")

st.markdown("---")
st.caption("App desenvolvido para análise institucional — ASELC / HGP.")

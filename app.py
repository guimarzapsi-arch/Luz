import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
import hashlib

# CONFIGURAÇÃO DE SEGURANÇA E INTERFACE
st.set_page_config(page_title="TopoLogos 4D | Matriz de Interferência", layout="wide")

# Autenticação Administrativa
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

if not st.session_state['autenticado']:
    st.title("Acesso Restrito - Espaço Clínico")
    senha = st.text_input("Senha Administrativa:", type="password")
    if st.button("Autenticar"):
        if senha == "admin123": # Altere conforme necessário
            st.session_state['autenticado'] = True
            st.rerun()
        else:
            st.error("Credenciais inválidas.")
    st.stop()

# PAINEL DE CONTROLE VISUAL (Estilo Construtor)
st.sidebar.title("Painel de Controle Estrutural")

st.sidebar.markdown("### 1. Estética e Identidade Visual")
tema_fundo = st.sidebar.radio("Fundo do Gráfico:", ["Branco Puro (Clínico)", "Preto (Profundo)"])
bg_color = 'white' if "Branco" in tema_fundo else 'black'
font_color = 'black' if "Branco" in tema_fundo else 'white'
escala_2d = st.sidebar.selectbox("Paleta da Fatia Temporal 2D:", ["Magma", "Viridis", "Inferno"])
escala_3d = st.sidebar.selectbox("Paleta da Estrutura 3D:", ["Greys", "Blues", "Purples"])

st.sidebar.markdown("### 2. Dados Clínicos e Criptografia")
significante = st.sidebar.text_input("Insira o Significante Parental/Clínico:", "Nome-do-Pai")
# Criptografia do significante para gerar a topologia sem expor os dados no motor matemático
hash_significante = hashlib.sha256(significante.encode('utf-8')).hexdigest()

st.sidebar.markdown("### 3. Neurofarmacologia e Física")
gaba = st.sidebar.slider("GABA (Inibição / Atenuação - γ):", 0.1, 3.0, 1.2)
glutamato = st.sidebar.slider("Glutamato (Amplitude - A):", 1.0, 5.0, 2.5)
dopamina = st.sidebar.slider("Dopamina (Velocidade/Freq - ω):", 1.0, 10.0, 4.0)

st.sidebar.markdown("### 4. Controle Holográfico (Tempo e Espaço)")
tempo_t = st.sidebar.slider("Momento da Sessão (Tempo - t):", 0.0, 20.0, 0.0, 0.1)
corte_z = st.sidebar.slider("Altura do Escorenate 2D (Eixo Z):", -4.0, 4.0, 0.0, 0.2)

# GERAÇÃO DA ARQUITETURA NEURAL 3D
@st.cache_data
def gerar_rede_neural():
    np.random.seed(42)
    def criar_lobo(centro, raios, n_pontos, nome):
        u, v = np.random.rand(n_pontos), np.random.rand(n_pontos)
        theta, phi = u * 2 * np.pi, np.arccos(2 * v - 1)
        r = np.cbrt(np.random.rand(n_pontos))
        x = centro[0] + raios[0] * r * np.sin(phi) * np.cos(theta)
        y = centro[1] + raios[1] * r * np.sin(phi) * np.sin(theta)
        z = centro[2] + raios[2] * r * np.cos(phi)
        return np.column_stack((x, y, z)), [nome]*n_pontos

    # Proporções cerebrais
    n_f, l_f = criar_lobo((0, 3, 2), (3, 2, 2), 200, "Córtex Pré-Frontal / Broca")
    n_t, l_t = criar_lobo((0, -1, -2), (4, 3, 2), 150, "Lobo Temporal / Wernicke")
    n_l, l_l = criar_lobo((0, 0, 0), (2, 2, 2), 150, "Sistema Límbico")
    
    nos = np.vstack((n_f, n_t, n_l))
    return nos, l_f + l_t + l_l

nos, labels = gerar_rede_neural()

# DETERMINAÇÃO DOS EPICENTROS (RSI) BASEADOS NO HASH DO SIGNIFICANTE
idx_R = int(hash_significante[0:4], 16) % len(nos)
idx_S = int(hash_significante[4:8], 16) % len(nos)
idx_I = int(hash_significante[8:12], 16) % len(nos)
focos = [nos[idx_R], nos[idx_S], nos[idx_I]]

# CÁLCULO 1: O BULK 3D (Cicatriz Estrutural / Inconsciente Atemporal)
I_3D = np.zeros(len(nos))
for foco in focos:
    dist = np.linalg.norm(nos - foco, axis=1)
    dist_segura = np.where(dist == 0, 0.1, dist)
    # Integral de Feynman colapsada na assíntota do trauma
    I_3D += (glutamato**2) / (dist_segura**2)

# Normalização para renderização
I_3D = np.clip(I_3D, 0, 15)

# CÁLCULO 2: A FRONTEIRA HOLOGRÁFICA 2D (O Tempo Presente)
malha_xy = np.linspace(-6, 6, 80)
X, Y = np.meshgrid(malha_xy, malha_xy)
Z_plano = np.full_like(X, corte_z)

Psi_2D = np.zeros_like(X)
k_onda = 2.0

for foco in focos:
    # Distância Euclidiana de cada ponto do plano até os epicentros 3D
    dist_plano = np.sqrt((X - foco[0])**2 + (Y - foco[1])**2 + (Z_plano - foco[2])**2)
    dist_plano = np.where(dist_plano == 0, 0.1, dist_plano)
    
    amortecimento = np.exp(-gaba * dist_plano * 0.3)
    fase_propagacao = np.cos(k_onda * dist_plano - dopamina * tempo_t)
    
    Psi_2D += glutamato * amortecimento * fase_propagacao

# RENDERIZAÇÃO COMPLEXA PLOTLY
fig = go.Figure()

# 1. Plotagem da Cicatriz 3D (A Estrutura de Base)
fig.add_trace(go.Scatter3d(
    x=nos[:, 0], y=nos[:, 1], z=nos[:, 2],
    mode='markers',
    marker=dict(
        size=I_3D * 1.5 + 2,
        color=I_3D,
        colorscale=escala_3d,
        opacity=0.4, # Semitransparente para representar o traço de memória
        showscale=False
    ),
    text=[f"{lbl}<br>Intensidade Base: {val:.2f}" for lbl, val in zip(labels, I_3D)],
    hoverinfo='text',
    name='Estrutura (Bulk)'
))

# 2. Plotagem do Plano Holográfico 2D (A Ondulação Temporal)
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z_plano,
    surfacecolor=Psi_2D,
    colorscale=escala_2d,
    opacity=0.85,
    cmin=-glutamato*1.5,
    cmax=glutamato*1.5,
    name='Fatia Temporal 2D',
    showscale=False
))

# 3. Destaque dos Focos do Real, Simbólico e Imaginário
fig.add_trace(go.Scatter3d(
    x=[f[0] for f in focos], y=[f[1] for f in focos], z=[f[2] for f in focos],
    mode='markers+text',
    text=['Real', 'Simbólico', 'Imaginário'],
    textposition="top center",
    marker=dict(size=12, color='red', symbol='diamond'),
    textfont=dict(color=font_color, family="Arial, sans-serif", size=14),
    name='Nós de Amarração'
))

fig.update_layout(
    title=dict(text=f"Projeção Topológica do Significante Holográfico", font=dict(color=font_color, size=24, family="Arial, sans-serif")),
    autosize=True,
    width=1200,
    height=800,
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        bgcolor=bg_color
    ),
    paper_bgcolor=bg_color,
    margin=dict(l=0, r=0, b=0, t=60)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
<div style="color: {font_color}; font-family: Arial, sans-serif; padding: 20px;">
A nuvem tridimensional fixa demonstra a <b>potenciação de longa duração</b> — a marca indelével do aprendizado e do trauma deixada pelo significante parental na arquitetura neurobiológica. O plano vibrante bidimensional, ajustável no eixo Z, representa a consciência e o discurso no momento analítico <b>t</b>, revelando como as ondas de neurotransmissores atravessam essa estrutura estática em tempo real.
</div>
""", unsafe_allow_html=True)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
import hashlib

# 1. CONFIGURAÇÃO DA INTERFACE
st.set_page_config(page_title="TopoLogos 4D | Neuropsicanálise e Ondas", layout="wide")
st.title("TopoLogos 4D: Matriz Neuropsicanalítica")
st.markdown("""
Representação tetradimensional (Espaço 3D + Tempo) do processamento do Significante. 
As ondas propagam-se através da anatomia topológica neural, moduladas por vetores psicanalíticos e farmacológicos.
""")

# 2. PAINEL DE CONTROLO CLÍNICO E FÍSICO
st.sidebar.header("Vetores Cronotopológicos")

significante = st.sidebar.text_input("Insira o Significante (Palavra):", "Angústia")
tempo = st.sidebar.slider("Tempo (4ª Dimensão - t):", 0.0, 10.0, 1.0, 0.1)

st.sidebar.header("Vetores Neurofarmacológicos")
st.sidebar.markdown("*Moduladores da Equação Ondulatória*")
gaba = st.sidebar.slider("GABA (Inibição / Amortecimento - γ):", 0.1, 5.0, 1.5)
glutamato = st.sidebar.slider("Glutamato (Excitação / Amplitude - A):", 0.1, 5.0, 2.0)
dopamina = st.sidebar.slider("Dopamina (Velocidade de Propagação - ω):", 1.0, 10.0, 5.0)
serotonina = st.sidebar.slider("Serotonina (Coerência de Fase - φ):", 0.0, 3.14, 0.0)

# 3. GERAÇÃO CIENTÍFICA DA ARQUITETURA NEURAL (NUVEM 3D)
@st.cache_data
def gerar_cerebro_matematico(resolucao=600):
    """Gera uma aproximação de rede neural baseada na morfologia lobar."""
    np.random.seed(42) # Mantém a estrutura anatómica constante
    
    def gerar_lobo(centro, raios, num_pontos, nome):
        u = np.random.rand(num_pontos)
        v = np.random.rand(num_pontos)
        theta = u * 2.0 * np.pi
        phi = np.arccos(2.0 * v - 1.0)
        r = np.cbrt(np.random.rand(num_pontos))
        
        x = centro[0] + raios[0] * r * np.sin(phi) * np.cos(theta)
        y = centro[1] + raios[1] * r * np.sin(phi) * np.sin(theta)
        z = centro[2] + raios[2] * r * np.cos(phi)
        
        return np.column_stack((x, y, z)), [nome]*num_pontos

    # Distribuição de neurónios por lobos funcionais
    frontal, c_f = gerar_lobo((0, 4, 2), (4, 3, 3), int(resolucao*0.3), 'Frontal (Linguagem/Broca)')
    parietal, c_p = gerar_lobo((0, 0, 5), (4, 3, 2), int(resolucao*0.2), 'Parietal (Sensorial)')
    occipital, c_o = gerar_lobo((0, -5, 1), (3, 2, 3), int(resolucao*0.15), 'Occipital (Visão)')
    temporal_esq, c_te = gerar_lobo((-4, -1, -1), (2, 4, 2), int(resolucao*0.1), 'Temporal Esq (Wernicke)')
    temporal_dir, c_td = gerar_lobo((4, -1, -1), (2, 4, 2), int(resolucao*0.1), 'Temporal Dir')
    limbico, c_l = gerar_lobo((0, 0, 0), (2, 2, 2), int(resolucao*0.15), 'Sistema Límbico (Pulsão/Emoção)')

    nos = np.vstack((frontal, parietal, occipital, temporal_esq, temporal_dir, limbico))
    labels = c_f + c_p + c_o + c_te + c_td + c_l
    return nos, labels

nos_neurais, labels_neurais = gerar_cerebro_matematico(800)

# 4. LINGUÍSTICA LACANIANA E MAPEAMENTO NEURAL
def calcular_foco_significante(texto, nos):
    """O significante atinge regiões cerebrais específicas baseado no seu peso estrutural."""
    hash_obj = hashlib.md5(texto.encode('utf-8')).hexdigest()
    # O hash determina o índice do nó "epicentro" do trauma/perturbação
    idx_real = int(hash_obj[0:8], 16) % len(nos)
    idx_simbolico = int(hash_obj[8:16], 16) % len(nos)
    idx_imaginario = int(hash_obj[16:24], 16) % len(nos)
    
    return nos[idx_real], nos[idx_simbolico], nos[idx_imaginario]

ep_R, ep_S, ep_I = calcular_foco_significante(significante, nos_neurais)

# 5. FÍSICA ONDULATÓRIA EM REDE NEURAL E NEUROFARMACOLOGIA
def calcular_propagacao_4D(nos, epicentro, t, amplitude, inibicao, vel, fase):
    """
    Equação da Onda Esférica Amortecida (4D):
    Z(r, t) = A * e^(-γ * r) * cos(k*r - ω*t + φ)
    """
    distancias = np.linalg.norm(nos - epicentro, axis=1)
    distancias_seguras = np.where(distancias == 0, 0.1, distancias)
    
    # K é o número de onda. ω (velocidade angular) é modulada pela Dopamina
    k = 2.0 
    
    termo_amortecimento = np.exp(-inibicao * distancias_seguras * 0.5)
    termo_oscilatorio = np.cos(k * distancias_seguras - vel * t + fase)
    
    # A intensidade da ativação neural
    intensidade = amplitude * termo_amortecimento * termo_oscilatorio
    return intensidade

# Superposição dos três registos (RSI) no cérebro perante o tempo t
int_R = calcular_propagacao_4D(nos_neurais, ep_R, tempo, glutamato, gaba, dopamina, serotonina)
int_S = calcular_propagacao_4D(nos_neurais, ep_S, tempo, glutamato, gaba, dopamina, serotonina)
int_I = calcular_propagacao_4D(nos_neurais, ep_I, tempo, glutamato, gaba, dopamina, serotonina)

Intensidade_Total = int_R + int_S + int_I

# 6. RENDERIZAÇÃO 4D DA ARQUITETURA NEURAL
# Criamos as arestas (sinapses) ligando apenas neurónios próximos (limiar de distância = 1.8)
matriz_dist = distance_matrix(nos_neurais, nos_neurais)
sinapses_x, sinapses_y, sinapses_z = [], [], []

# Otimização matemática para desenhar a rede sem sobrecarregar o renderizador
limiar_conexao = 1.5
for i in range(len(nos_neurais)):
    conexoes = np.where((matriz_dist[i] > 0) & (matriz_dist[i] < limiar_conexao))[0]
    # Limitar o número de ligações por neurónio para clareza visual
    for j in conexoes[:3]: 
        sinapses_x.extend([nos_neurais[i, 0], nos_neurais[j, 0], None])
        sinapses_y.extend([nos_neurais[i, 1], nos_neurais[j, 1], None])
        sinapses_z.extend([nos_neurais[i, 2], nos_neurais[j, 2], None])

fig = go.Figure()

# Camada 1: Estrutura Sináptica (As vias de escoamento da pulsão)
fig.add_trace(go.Scatter3d(
    x=sinapses_x, y=sinapses_y, z=sinapses_z,
    mode='lines',
    line=dict(color='rgba(255, 255, 255, 0.05)', width=1),
    hoverinfo='none'
))

# Camada 2: Nós Neurais Ativados (As representações/significantes)
fig.add_trace(go.Scatter3d(
    x=nos_neurais[:, 0],
    y=nos_neurais[:, 1],
    z=nos_neurais[:, 2],
    mode='markers',
    marker=dict(
        size=np.abs(Intensidade_Total) * 3 + 2, # O tamanho reflete o nível de excitação local
        color=Intensidade_Total,
        colorscale='Plasma',
        opacity=0.8,
        colorbar=dict(title="Carga Pulsional (mV)"),
        cmin=-glutamato*2,
        cmax=glutamato*2
    ),
    text=[f"Região: {lbl}<br>Excitação: {val:.2f}" for lbl, val in zip(labels_neurais, Intensidade_Total)],
    hoverinfo='text'
))

fig.update_layout(
    title=f"Ressonância Neural do Significante: '{significante}' no Tempo t={tempo}s",
    autosize=True,
    width=1100,
    height=800,
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showgrid=False, zeroline=False, visible=False),
        bgcolor='black'
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Análise Científica:**  
A matriz acima é um modelo biológico tetradimensional aproximado. O **Tempo (t)** atua como a fase oscilatória da onda[span_2](start_span)[span_2](end_span). 
Quando altera a barra de Tempo, está a visualizar a propagação da energia psíquica e elétrica através dos tratos neurais, partindo 
dos nós do Real, Simbólico e Imaginário instigados pelo Significante inserido[span_3](start_span)[span_3](end_span). Medicamentos excitatórios amplificam a ressonância global, enquanto inibidores (como o GABA) forçam o decaimento rápido do sinal no espaço[span_4](start_span)[span_4](end_span).
""")

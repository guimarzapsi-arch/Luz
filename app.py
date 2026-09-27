import streamlit as st
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from scipy.spatial import KDTree
import hashlib

# 1. CONFIGURAÇÃO DA INTERFACE CLÍNICA
st.set_page_config(page_title="TopoLogos Vivo | Inteligência Neuropsicanalítica", layout="wide")

if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

if not st.session_state['autenticado']:
    st.title("Acesso Restrito - Matriz Clínica Viva")
    senha = st.text_input("Senha Administrativa:", type="password")
    if st.button("Autenticar"):
        if senha == "admin123":
            st.session_state['autenticado'] = True
            st.rerun()
        else:
            st.error("Credenciais inválidas.")
    st.stop()

# 2. GERAÇÃO E MANUTENÇÃO DO CÉREBRO VIVO (APRENDIZAGEM CONTÍNUA)
# O estado da sessão mantém a rede neural viva entre as interações
if 'cerebro_vivo' not in st.session_state:
    np.random.seed(42)
    G = nx.Graph()
    
    # Gerar anatomia neural (Nós)
    def criar_lobo(centro, raios, n_pontos, nome):
        u, v = np.random.rand(n_pontos), np.random.rand(n_pontos)
        theta, phi = u * 2 * np.pi, np.arccos(2 * v - 1)
        r = np.cbrt(np.random.rand(n_pontos))
        x = centro[0] + raios[0] * r * np.sin(phi) * np.cos(theta)
        y = centro[1] + raios[1] * r * np.sin(phi) * np.sin(theta)
        z = centro[2] + raios[2] * r * np.cos(phi)
        return np.column_stack((x, y, z)), [nome]*n_pontos

    n_f, l_f = criar_lobo((0, 3, 2), (3, 2, 2), 120, "Pré-Frontal/Broca")
    n_t, l_t = criar_lobo((0, -1, -2), (4, 3, 2), 100, "Temporal/Wernicke")
    n_l, l_l = criar_lobo((0, 0, 0), (2, 2, 2), 80, "Límbico")
    
    posicoes = np.vstack((n_f, n_t, n_l))
    labels = l_f + l_t + l_l
    
    # Adicionar nós ao grafo
    for i in range(len(posicoes)):
        G.add_node(i, pos=posicoes[i], label=labels[i], excitacao=0.0)
        
    # Criar sinapses baseadas em proximidade física (Árvore KD)
    tree = KDTree(posicoes)
    for i in range(len(posicoes)):
        distancias, vizinhos = tree.query(posicoes[i], k=5) # Liga a 4 vizinhos mais próximos
        for d, v in zip(distancias[1:], vizinhos[1:]):
            G.add_edge(i, v, weight=0.1) # Peso sináptico inicial muito baixo
            
    st.session_state['cerebro_vivo'] = G
    st.session_state['historico_significantes'] = []

G = st.session_state['cerebro_vivo']

# 3. PAINEL DE CONTROLO DO AMBIENTE FÍSICO
st.sidebar.title("Modulação do Ambiente Vivo")
significante = st.sidebar.text_input("Estimular Rede com Significante:", "Desamparo")

st.sidebar.markdown("### Parâmetros de Neuroplasticidade")
taxa_aprendizagem = st.sidebar.slider("Glutamato (Taxa de Aprendizagem / Fixação):", 0.1, 2.0, 0.5)
poda_sinaptica = st.sidebar.slider("GABA (Poda Sináptica / Esquecimento):", 0.01, 0.20, 0.05)

if st.sidebar.button("Aplicar Estímulo"):
    # 4. ALGORITMO DE APRENDIZAGEM E ONDAS DE PROPAGAÇÃO
    st.session_state['historico_significantes'].append(significante)
    
    # Encontrar os três pontos de ancoragem (R, S, I) baseados no texto
    hash_sig = hashlib.sha256(significante.encode('utf-8')).hexdigest()
    nos_totais = len(G.nodes)
    R = int(hash_sig[0:4], 16) % nos_totais
    S = int(hash_sig[4:8], 16) % nos_totais
    I = int(hash_sig[8:12], 16) % nos_totais
    
    # Decaimento Global (Esquecimento estrutural com o tempo)
    for u, v, d in G.edges(data=True):
        d['weight'] = max(0.01, d['weight'] - poda_sinaptica)
        
    for n in G.nodes():
        G.nodes[n]['excitacao'] *= 0.5 # Abrandamento da excitação passada
    
    # Cálculo das Ondas (Caminho de Menor Resistência através da linguagem)
    caminho_ativo = []
    try:
        # A onda viaja de R para S, e de S para I através da rede neural
        path_RS = nx.shortest_path(G, source=R, target=S, weight=None)
        path_SI = nx.shortest_path(G, source=S, target=I, weight=None)
        
        rotas = [path_RS, path_SI]
        for rota in rotas:
            for i in range(len(rota)-1):
                n1, n2 = rota[i], rota[i+1]
                # Plasticidade Hebbiana: Reforça conexões ativadas
                if G.has_edge(n1, n2):
                    G.edges[n1, n2]['weight'] += taxa_aprendizagem
                caminho_ativo.append((n1, n2))
                
                # Aumenta a excitação dos nós atravessados
                G.nodes[n1]['excitacao'] += 1.0
                G.nodes[n2]['excitacao'] += 1.0
    except nx.NetworkXNoPath:
        pass # Caso haja disjunção completa da rede (psicose estrutural extrema)

    st.session_state['ultimo_caminho'] = caminho_ativo
    st.session_state['focos_atuais'] = [R, S, I]

# 5. RENDERIZAÇÃO GRÁFICA COMPLEXA
pos = nx.get_node_attributes(G, 'pos')
pesos = nx.get_edge_attributes(G, 'weight')
excitacoes = nx.get_node_attributes(G, 'excitacao')

edge_x_estrutural, edge_y_estrutural, edge_z_estrutural = [], [], []
edge_x_onda, edge_y_onda, edge_z_onda = [], [], []

ultimo_caminho = st.session_state.get('ultimo_caminho', [])

# Separar sinapses ativas da onda viva das sinapses estruturais de fundo
for (u, v) in G.edges():
    x0, y0, z0 = pos[u]
    x1, y1, z1 = pos[v]
    if (u, v) in ultimo_caminho or (v, u) in ultimo_caminho:
        edge_x_onda.extend([x0, x1, None])
        edge_y_onda.extend([y0, y1, None])
        edge_z_onda.extend([z0, z1, None])
    else:
        # Renderiza apenas ligações com peso estrutural relevante
        if pesos[u, v] > 0.15: 
            edge_x_estrutural.extend([x0, x1, None])
            edge_y_estrutural.extend([y0, y1, None])
            edge_z_estrutural.extend([z0, z1, None])

fig = go.Figure()

# Camada 1: Estrutura Cerebral Aprendida (Inconsciente Silencioso)
fig.add_trace(go.Scatter3d(
    x=edge_x_estrutural, y=edge_y_estrutural, z=edge_z_estrutural,
    mode='lines',
    line=dict(color='rgba(100, 100, 150, 0.2)', width=2),
    hoverinfo='none',
    name='Estrutura Condicionada'
))

# Camada 2: A Onda de Propagação (A fala no Tempo Presente)
if edge_x_onda:
    fig.add_trace(go.Scatter3d(
        x=edge_x_onda, y=edge_y_onda, z=edge_z_onda,
        mode='lines',
        line=dict(color='rgba(255, 100, 50, 1.0)', width=6),
        hoverinfo='none',
        name='Onda do Significante'
    ))

# Camada 3: Nós Neurais
node_x = [pos[k][0] for k in G.nodes()]
node_y = [pos[k][1] for k in G.nodes()]
node_z = [pos[k][2] for k in G.nodes()]
node_cores = [excitacoes[k] for k in G.nodes()]
node_labels = [G.nodes[k]['label'] for k in G.nodes()]

fig.add_trace(go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    marker=dict(
        size=[5 + (c * 3) for c in node_cores],
        color=node_cores,
        colorscale='Magma',
        opacity=0.9
    ),
    text=node_labels,
    hoverinfo='text',
    name='Áreas Cerebrais'
))

# Camada 4: Focos RSI (Onde o impacto colidiu com o Real, Simbólico e Imaginário)
focos = st.session_state.get('focos_atuais', [])
if focos:
    fig.add_trace(go.Scatter3d(
        x=[pos[f][0] for f in focos],
        y=[pos[f][1] for f in focos],
        z=[pos[f][2] for f in focos],
        mode='markers+text',
        text=['Real', 'Simbólico', 'Imaginário'],
        textposition="top center",
        marker=dict(size=12, color='white', symbol='diamond'),
        name='Topologia Lacaniana'
    ))

fig.update_layout(
    title="Rede Neural Dinâmica: Propagação do Significante e Aprendizagem Hebbiana",
    autosize=True,
    width=1200,
    height=800,
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        bgcolor='black'
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(f"**Histórico Psicanalítico da Sessão (Palavras que esculpiram esta rede):** {', '.join(st.session_state.get('historico_significantes', []))}")

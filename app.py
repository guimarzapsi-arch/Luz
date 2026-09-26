import streamlit as st
import numpy as np
import plotly.graph_objects as go
import hashlib

# 1. CONFIGURAÇÃO DA INTERFACE E ESTÉTICA
st.set_page_config(page_title="TopoLogos | RSI e Ondas", layout="wide")

st.title("TopoLogos: A Ótica do Inconsciente")
st.markdown("""
Esta plataforma traduz a estrutura do sujeito em uma representação matemática e ondulatória. 
O **Significante** (linguística) atua como um perturbador de frequências, moldando o padrão 
de interferência entre três fontes de ondas luminosas: o **Real**, o **Simbólico** e o **Imaginário** (Psicanálise Lacaniana).
""")

# 2. PAINEL DE CONTROLE VISUAL (Barra Lateral)
st.sidebar.header("Painel de Modulação")
st.sidebar.markdown("Ajuste a representação sem precisar programar.")

significante = st.sidebar.text_input("Insira o Significante (Palavra):", "Luz")

escala_cor = st.sidebar.selectbox(
    "Paleta de Cores (Espectro de Luz):",
    ["Magma", "Viridis", "Plasma", "Inferno", "Cividis"]
)

amortecimento = st.sidebar.slider("Resistência do Meio (Amortecimento):", 0.1, 3.0, 1.0)
resolucao = st.sidebar.slider("Resolução da Malha Topológica:", 50, 150, 100)

# 3. LÓGICA LINGUÍSTICA -> MATEMÁTICA
def significante_para_frequencias(texto):
    """
    Converte a palavra do usuário em frequências baseadas em criptografia (hash).
    Garante que a mesma palavra sempre gere a mesma topologia exata.
    """
    hash_obj = hashlib.md5(texto.encode('utf-8'))
    hex_digest = hash_obj.hexdigest()
    
    # Extrai 3 frequências únicas a partir dos fragmentos do hash
    freq_R = (int(hex_digest[0:4], 16) % 15) + 2  # Frequência do Real
    freq_S = (int(hex_digest[4:8], 16) % 15) + 2  # Frequência do Simbólico
    freq_I = (int(hex_digest[8:12], 16) % 15) + 2 # Frequência do Imaginário
    
    return freq_R, freq_S, freq_I

f_R, f_S, f_I = significante_para_frequencias(significante)

st.sidebar.markdown("---")
st.sidebar.markdown("**Frequências Geradas pelo Significante:**")
st.sidebar.write(f"- Freq (Real): {f_R} Hz")
st.sidebar.write(f"- Freq (Simbólico): {f_S} Hz")
st.sidebar.write(f"- Freq (Imaginário): {f_I} Hz")

# 4. FÍSICA ONDULATÓRIA (MECÂNICA E SUPERPOSIÇÃO)
# Criação do espaço tridimensional (O lugar do Outro)
x = np.linspace(-10, 10, resolucao)
y = np.linspace(-10, 10, resolucao)
X, Y = np.meshgrid(x, y)

def onda_esferica(X, Y, x_origem, y_origem, frequencia, amortecimento):
    """
    Equação de uma onda esférica: Z = A * sin(k*r - w*t) / r
    A amplitude cai conforme a distância (amortecimento).
    """
    distancia = np.sqrt((X - x_origem)**2 + (Y - y_origem)**2)
    # Evitar divisão por zero somando um epsilon
    distancia_segura = np.where(distancia == 0, 0.1, distancia)
    
    # Onda transversal de luz
    onda = np.sin(frequencia * distancia_segura) / (distancia_segura * amortecimento)
    return onda

# Posições nodais formando um triângulo (analogia ao Nó Borromeano)
# Se um ponto ceder, a interferência global colapsa.
Z_Real = onda_esferica(X, Y, -3, -3, f_R, amortecimento)
Z_Simbolico = onda_esferica(X, Y, 3, -3, f_S, amortecimento)
Z_Imaginario = onda_esferica(X, Y, 0, 3, f_I, amortecimento)

# O Sujeito Lacaniano como o Princípio da Superposição
Z_Sujeito = Z_Real + Z_Simbolico + Z_Imaginario

# 5. RENDERIZAÇÃO TOPOLÓGICA 3D
fig = go.Figure(data=[go.Surface(
    z=Z_Sujeito,
    x=X,
    y=Y,
    colorscale=escala_cor,
    showscale=False
)])

# Configurações do ambiente visual e da luz
fig.update_layout(
    title=f"Topologia de Interferência para o significante: '{significante}'",
    autosize=True,
    width=1000,
    height=700,
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis_title='Vetor X',
        yaxis_title='Vetor Y',
        zaxis_title='Amplitude da Pulsão',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showgrid=False, zeroline=False, visible=False),
        bgcolor='black' # Fundo escuro transcendental
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
*Rigor Científico Aplicado:* O gráfico acima não é uma animação aleatória. Ele resolve as equações diferenciais 
de superposição de ondas em tempo real. A palavra digitada altera o *hash* criptográfico, que define a 
frequência do vetor de onda ($k$). Os picos e vales representam as intensidades de interferência construtiva e 
destrutiva, criando a topologia metafórica onde os três registros se atravessam.
""")

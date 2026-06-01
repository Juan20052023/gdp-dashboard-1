import streamlit as st
import math
import copy
from scipy.optimize import linprog
import networkx as nx
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Branch & Bound", layout="wide")
st.title("🌳 Optimizador Lineal: Branch & Bound")
st.markdown("Resuelve problemas de 3 variables mostrando el paso a paso de la ramificación y una visualización gráfica del árbol completo.")

# --- SECCIÓN DE ENTRADA DE DATOS ---
st.header("1. Definir el Problema")

col_opt, col_blank = st.columns([1, 3])
with col_opt:
    tipo_opt = st.selectbox("Objetivo:", ["Maximizar", "Minimizar"])

st.subheader("Función Objetivo (Z)")
c1_z, c2_z, c3_z = st.columns(3)
c1 = c1_z.number_input("Coeficiente x1", value=0.0, step=1.0)
c2 = c2_z.number_input("Coeficiente x2", value=0.0, step=1.0)
c3 = c3_z.number_input("Coeficiente x3", value=0.0, step=1.0)

st.subheader("Restricciones (Sujeto a:)")
A_inputs = []
b_inputs = []
signos = []

# Crear 3 filas para las restricciones
for i in range(3):
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 2])
    with col1: a1 = st.number_input(f"x1 (R{i+1})", value=0.0, step=1.0, key=f"a1_{i}")
    with col2: a2 = st.number_input(f"x2 (R{i+1})", value=0.0, step=1.0, key=f"a2_{i}")
    with col3: a3 = st.number_input(f"x3 (R{i+1})", value=0.0, step=1.0, key=f"a3_{i}")
    with col4: signo = st.selectbox("Signo", ["<=", ">="], key=f"sign_{i}")
    with col5: b = st.number_input(f"Lado Derecho (R{i+1})", value=0.0, step=1.0, key=f"b_{i}")
    
    A_inputs.append([a1, a2, a3])
    signos.append(signo)
    b_inputs.append(b)

st.markdown("---")

# --- MOTOR DE CÁLCULO (BRANCH & BOUND) ---
if st.button("🚀 Ejecutar Algoritmo", type="primary"):
    st.header("2. Historial de Ramificación")
    
    consola = []
    tree = nx.DiGraph()

    es_max = (tipo_opt == "Maximizar")
    c_scipy = [-x for x in [c1, c2, c3]] if es_max else [c1, c2, c3]

    A_inicial = []
    b_inicial = []
    
    for i in range(3):
        row_A = A_inputs[i]
        rhs = b_inputs[i]
        if signos[i] == ">=":
            row_A = [-x for x in row_A]
            rhs = -rhs
        if any(val != 0 for val in row_A) or rhs != 0:
            A_inicial.append(row_A)
            b_inicial.append(rhs)

    mejor_z_entero = -float('inf') if es_max else float('inf')
    mejor_solucion_entera = None
    
    nodos_pila = [{"id": 0, "A": A_inicial, "b": b_inicial, "historia": "Nodo 0 (Raíz)", "parent": None, "edge_label": ""}]
    contador_nodos = 0

    while nodos_pila:
        nodo = nodos_pila.pop()
        consola.append(f"-> Explorando {nodo['historia']}...")

        # Añadir estructura básica al grafo
        tree.add_node(nodo['id'], label=f"P_{nodo['id']}", color="#ffffff") 
        if nodo['parent'] is not None:
            tree.add_edge(nodo['parent'], nodo['id'], label=nodo['edge_label'])

        res = linprog(c_scipy, A_ub=nodo['A'], b_ub=nodo['b'], bounds=(0, None), method='highs')

        # CASO 1: INFACTIBLE
        if not res.success:
            consola.append("   [x] Podado: Problema Infactible.\n")
            tree.nodes[nodo['id']]['color'] = "#ff6666" # Rojo
            tree.nodes[nodo['id']]['label'] = f"P_{nodo['id']}\nInfactible"
            continue

        z_actual = -res.fun if es_max else res.fun
        x_val = res.x

        # CASO 2: PODADO POR COTA (Ya tenemos una solución entera mejor)
        if (es_max and z_actual <= mejor_z_entero) or (not es_max and z_actual >= mejor_z_entero):
            consola.append(f"   [x] Podado por Cota: Z={z_actual:.2f} es peor o igual que mejor Z={mejor_z_entero:.2f}\n")
            tree.nodes[nodo['id']]['color'] = "#ffb366" # Naranja
            tree.nodes[nodo['id']]['label'] = f"P_{nodo['id']}\nZ={z_actual:.2f}\nPodado"
            continue

        consola.append(f"   Resultado: Z = {z_actual:.4f} | Variables: [x1={x_val[0]:.4f}, x2={x_val[1]:.4f}, x3={x_val[2]:.4f}]")

        # Verificar decimales
        idx_fraccionario = -1
        valor_fraccionario = 0
        for i, val in enumerate(x_val):
            if abs(val - round(val)) > 1e-5:
                idx_fraccionario = i
                valor_fraccionario = val
                break

        # CASO 3: SOLUCIÓN ENTERA ENCONTRADA
        if idx_fraccionario == -1:
            consola.append(f"   [!] ¡Solución Entera Encontrada! Actualizando mejor Z a {z_actual:.4f}\n")
            mejor_z_entero = z_actual
            mejor_solucion_entera = x_val
            
            tree.nodes[nodo['id']]['color'] = "#66ff66" # Verde
            tree.nodes[nodo['id']]['label'] = f"P_{nodo['id']}\nZ={z_actual:.2f}\nEntero"
            continue

        # CASO 4: FRACCIONARIO (RAMIFICAR)
        var_nombre = f"x{idx_fraccionario + 1}"
        consola.append(f"   [*] Ramificando en variable {var_nombre} = {valor_fraccionario:.4f}\n")
        
        tree.nodes[nodo['id']]['color'] = "#66b3ff" # Azul claro
        tree.nodes[nodo['id']]['label'] = f"P_{nodo['id']}\nZ={z_actual:.2f}"

        # --- Creación de hijos ---
        # Rama Izquierda (x <= floor)
        A_izq = copy.deepcopy(nodo['A'])
        b_izq = copy.deepcopy(nodo['b'])
        nueva_fila_izq = [0, 0, 0]; nueva_fila_izq[idx_fraccionario] = 1
        A_izq.append(nueva_fila_izq)
        b_izq.append(math.floor(valor_fraccionario))
        
        contador_nodos += 1
        nodos_pila.append({
            "id": contador_nodos, "A": A_izq, "b": b_izq, 
            "historia": f"Nodo {contador_nodos} (De nodo {nodo['id']}, {var_nombre} <= {math.floor(valor_fraccionario)})",
            "parent": nodo['id'], "edge_label": rf"${var_nombre} \leq {math.floor(valor_fraccionario)}$"
        })

        # Rama Derecha (x >= ceil) -> -x <= -ceil
        A_der = copy.deepcopy(nodo['A'])
        b_der = copy.deepcopy(nodo['b'])
        nueva_fila_der = [0, 0, 0]; nueva_fila_der[idx_fraccionario] = -1
        A_der.append(nueva_fila_der)
        b_der.append(-math.ceil(valor_fraccionario))

        contador_nodos += 1
        nodos_pila.append({
            "id": contador_nodos, "A": A_der, "b": b_der, 
            "historia": f"Nodo {contador_nodos} (De nodo {nodo['id']}, {var_nombre} >= {math.ceil(valor_fraccionario)})",
            "parent": nodo['id'], "edge_label": rf"${var_nombre} \geq {math.ceil(valor_fraccionario)}$"
        })

    # --- IMPRIMIR HISTORIAL Y LEYENDA ---
    st.code("\n".join(consola), language="bash")

    # --- VISUALIZACIÓN GRÁFICA DEL ÁRBOL ---
    st.header("3. Visualización Gráfica del Árbol")
    
    st.markdown("""
    **Leyenda del Grafo:**
    * 🔵 **Azul:** Nodo ramificado (tiene decimales).
    * 🟢 **Verde:** Solución Entera factible (la mejor encontrada prevalece).
    * 🔴 **Rojo:** Nodo Infactible (no tiene solución).
    * 🟠 **Naranja:** Podado por cota (tiene solución, pero es peor que una solución entera ya encontrada).
    """)

    if len(tree.nodes()) > 0:
        
        def calculate_positions(graph, root_node, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, level=0):
            if pos is None:
                pos = {root_node: (xcenter, vert_loc)}
            else:
                pos[root_node] = (xcenter, vert_loc)
                
            neighbors = list(graph.successors(root_node))
            if len(neighbors) != 0:
                dx = width / 2
                left_x = xcenter - dx / 2
                next_y = vert_loc - vert_gap
                calculate_positions(graph, neighbors[0], width=dx, vert_gap=vert_gap, vert_loc=next_y, xcenter=left_x, pos=pos, level=level+1)
                if len(neighbors) > 1:
                    right_x = xcenter + dx / 2
                    calculate_positions(graph, neighbors[1], width=dx, vert_gap=vert_gap, vert_loc=next_y, xcenter=right_x, pos=pos, level=level+1)
            return pos

        if 0 in tree.nodes():
            try:
                pos = calculate_positions(tree, 0)
                y_max = max([p[1] for p in pos.values()])
                for node in pos:
                    x, y = pos[node]
                    pos[node] = (x, y_max - y)

                fig, ax = plt.subplots(figsize=(12, 9))
                
                # Extraer atributos para dibujar
                node_colors = [tree.nodes[n].get('color', 'white') for n in tree.nodes()]
                node_labels = nx.get_node_attributes(tree, 'label')
                edge_labels = nx.get_edge_attributes(tree, 'label')

                # Dibujar
                nx.draw_networkx_nodes(tree, pos, ax=ax, node_size=3500, node_color=node_colors, edgecolors='black')
                nx.draw_networkx_labels(tree, pos, labels=node_labels, ax=ax, font_size=11, font_family='sans-serif')
                nx.draw_networkx_edges(tree, pos, ax=ax, arrowstyle='-|>', arrowsize=15, width=1.5)
                nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, ax=ax, font_size=12, label_pos=0.5, rotate=False, font_family='serif')

                # Añadir un pequeño margen para que los círculos no se corten en los bordes
                ax.margins(0.1)
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error al dibujar el grafo: {e}")
        else:
            st.info("No se generaron nodos válidos.")

    # --- SOLUCIÓN FINAL ---
    st.header("4. Solución Final")
    if mejor_solucion_entera is not None:
        st.success(f"**Mejor Z Entero Encontrado:** {mejor_z_entero:.4f}")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("x1", round(mejor_solucion_entera[0]))
        col_res2.metric("x2", round(mejor_solucion_entera[1]))
        col_res3.metric("x3", round(mejor_solucion_entera[2]))
    else:
        st.error("No se encontró ninguna solución entera factible para este problema.")
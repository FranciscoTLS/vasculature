#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from .masks import sausage_mask
from .helpers import point_line_distance, get_vector_length, get_vector, get_index, get_position
from .physics import update_radius, get_radius

def generate_mask(N_mesh, dL, e, nodes, segments, **kwargs):
    # Gera a região onde haverá vascularização dentro da malha
    # Utiliza máscaras de "salsicha"
    mask = np.ones(N_mesh)
    for segment in segments:
        mask *= sausage_mask(N_mesh, 
                             dL, 
                             np.transpose([get_position(nodes[node_index], dL) 
                                           for node_index in segment]), 
                             e)
    return mask

def roll_terminal_node(probability_mask):
    # Seleciona aleatoriamente um nó terminal com base na máscara de probabilidade fornecida.
    # node_index -> [x, y, z] (indices)
    shape = probability_mask.shape
    size = np.prod(shape)
    nodes = np.random.choice(np.arange(size), 1, False, probability_mask.flatten())
    node_index = tuple(np.vstack(np.unravel_index(nodes, shape)).T[0])
    return node_index

def find_closest_segment(node, nodes, segments):
    # Retorna o indice do segmento mais próximo ao nó fornecido.
    min_distance = 0
    closest_segment_index = None
    for segment_index in range(len(segments)):
        distance = point_line_distance(node, *[nodes[node_index] 
                                               for node_index in segments[segment_index]])
        if distance < min_distance or not min_distance:
            min_distance = distance
            closest_segment_index = segment_index
    return closest_segment_index

def optimize_join(nodes, P, F):
    # nodes -> [start_node, end_node, term_node] -> Index!
    # P -> [P_start, P_end, P_term]
    # F -> [F_start, F_end, F_branch]
    
    best_join = [0, None, None, None, None]
    
    min_range = np.amin(nodes, axis=0)
    max_range = np.amax(nodes, axis=0) + 1
    dimensions = len(min_range)
    index = [None]*dimensions
    
    def iterator(dimension=0):
        range_start = min_range[dimension]
        range_end = max_range[dimension]
        if dimension == dimensions - 1:
            for i in range(range_start, range_end):
                index[dimension] = i
                L = [get_vector_length(get_vector(nodes[j], index)) for j in range(3)]
                P_join = P[0] + L[0]*(P[1] - P[0])/(L[0] + L[1])
                dP = [abs(P_join - P[0]), abs(P[1] - P_join), abs(P[2] - P_join)]
                if 0 not in L:
                    volume = np.sum(np.multiply(np.sqrt(np.divide(F, dP)), np.power(L, 1.5)))
                    if volume < best_join[0] or not best_join[0]:
                        best_join[:] = [volume, tuple(index), P_join, L, dP]
        else:
            for i in range(range_start, range_end):
                index[dimension] = i
                iterator(dimension + 1)
    
    iterator()
    return best_join[1:]

def update_parents(segment_index, parents, flows, radii, extra_flow):
    # Atualiza as dependências das listas de propriedades de acordo com a hierarquia dos segmentos
    
    def recursive(child_index):
        parent_index = parents[child_index]
        if parent_index is not None:
            radii[parent_index] = update_radius(radii[parent_index], 
                                                flows[parent_index], 
                                                extra_flow)
            flows[parent_index] += extra_flow
            recursive(parent_index)
    
    recursive(segment_index)

def add_new_terminal(term_node, P_term, F_term, nodes, segments, parents, children, pressures, flows, radii, **kwargs):
    # term_node -> Index!
    
    closest_segment_index = find_closest_segment(term_node, nodes, segments)
    
    start_node_index, end_node_index = segments[closest_segment_index]
    term_node_index = len(nodes)
    join_node_index = term_node_index + 1
    
    end_segment_index = len(segments)
    branch_segment_index = end_segment_index + 1
    
    # Achar o ponto ideal do nó de conexão!
    # Nós:
    P_start = pressures[start_node_index]
    P_end = pressures[end_node_index]
    # Segmentos:
    F_end = flows[closest_segment_index]
    F_branch = F_term
    F_start = F_end + F_branch
    
    join_node, P_join, lengths, dP = optimize_join([nodes[start_node_index], nodes[end_node_index], term_node], 
                                                   [P_start, P_end, P_term], 
                                                   [F_start, F_end, F_branch])
    
    nodes.append(term_node)
    nodes.append(join_node)
    pressures.append(P_term)
    pressures.append(P_join)
    
    segments[closest_segment_index][1] = join_node_index
    segments.append([join_node_index, end_node_index])
    segments.append([join_node_index, term_node_index])
    parents.append(closest_segment_index)
    parents.append(closest_segment_index)
    for child_index in children[closest_segment_index]:
        if child_index is not None:
            parents[child_index] = end_segment_index
    children.append(children[closest_segment_index])
    children.append([None, None])
    children[closest_segment_index] = [end_segment_index, branch_segment_index]
    flows[closest_segment_index] = F_start
    flows.append(F_end)
    flows.append(F_branch)
    radii[closest_segment_index] = get_radius(F_start, lengths[0], dP[0])
    radii.append(get_radius(F_end, lengths[1], dP[1]))
    radii.append(get_radius(F_branch, lengths[2], dP[2]))
    
    update_parents(closest_segment_index, parents, flows, radii, F_branch)

def initialize_mesh(N_mesh, L):
    dL = [L[i]/(N_mesh[i] - 1) for i in range(len(N_mesh))]
    mesh = np.ones(N_mesh)
    return mesh, dL
    
def initialize_params(M, dL, X_in, X_out, X_term, P_in, P_out, P_term, f_term, **kwargs):
    
    dV = np.prod(dL)
    F_term = f_term*dV
    
    params = {
        'M': M,
        'N_mesh': M.shape,
        'dL': dL,
        'P_term': P_term,
        'F_term': F_term,
        'in': {
            'nodes': [get_index(X_in, dL), get_index(X_term, dL)],
            'pressures': [P_in, P_term],
            'segments': [[0, 1]],
            'parents': [None],
            'children': [[None, None]],
            'flows': [F_term],
            'radii': [get_radius(F_term, get_vector_length(get_vector(X_in, X_term)), P_in - P_term)]
        },
        'out': {
            'nodes': [get_index(X_out, dL), get_index(X_term, dL)],
            'pressures': [P_out, P_term],
            'segments': [[0, 1]],
            'parents': [None],
            'children': [[None, None]],
            'flows': [F_term],
            'radii': [get_radius(F_term, get_vector_length(get_vector(X_out, X_term)), P_term - P_out)]
        }
    }
    return params

def iterate_model(iterations, params, seed=None, track_progress=False):
    
    if seed:
        np.random.seed(seed) # Inicialização do randomizador
        
    N_term = 0
    M = params['M']
    N_mesh = params['N_mesh']
    dL = params['dL']
    P_term = params['P_term']
    F_term = params['F_term']
    params_in = params['in']
    params_out = params['out']

    mask_in = generate_mask(N_mesh, dL, **params_in, e=0.2)
    mask_out = generate_mask(N_mesh, dL, **params_out, e=0.2)
        
    print('Iterando...', end='\r')
    for i in range(iterations):
        # Acompanhamento do progresso
        if track_progress:
            print('Iterando... (%d/%d)'%(N_term, iterations), end='\r')
        
        # Atualização da máscara de probabilidade
        M_i = M*mask_in*mask_out
        nodes_available = np.sum(M_i)

        # Caso ainda haja espaço na malha para novas regiões terminais:
        if nodes_available != 0:
            probability_mask = M_i/nodes_available # Probabilidade de terminais da malha

            new_terminal = roll_terminal_node(probability_mask) # Nova região terminal

            # Atualização das listas de dados
            add_new_terminal(new_terminal, P_term, F_term, **params_in)
            add_new_terminal(new_terminal, P_term, F_term, **params_out)
            N_term += 1

            # Atualização dos componentes da máscara de probabilidade
            mask_in = generate_mask(N_mesh, dL, **params_in, e=0.2)
            mask_out = generate_mask(N_mesh, dL, **params_out, e=0.2)
        else:
            print('Não cabem mais pontos terminais na malha!\n Aumente a resolução ou reduza a distância mínima entre novas regiões terminais e vasos já existentes!')
            i = iterations - 1
    print('Regiões terminais adicionadas: %d'%(N_term))


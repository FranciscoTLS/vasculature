#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

def get_position(index, h):
    return np.multiply(index, h)

def get_index(position, h):
    return tuple(np.divide(position, h).astype(int))

def get_vector(p_1, p_2):
    return np.subtract(p_2, p_1)

def get_vector_length(v):
    return sqrt(np.dot(v, v))

def projection_scalar(u, v):
    # Projection of u onto v
    return np.dot(v, u)/np.dot(v, v)

def point_line_distance(p, l_1, l_2):
    line_vector = get_vector(l_1, l_2)
    point_vector = get_vector(l_1, p)
    if line_vector.any():
        scalar = projection_scalar(point_vector, line_vector)
    else:
        scalar = -1
    if scalar <= 0:
        return get_vector_length(point_vector)
    elif scalar < 1:
        projection = l_1 + scalar*line_vector
        return get_vector_length(get_vector(projection, p))
    else:
        return get_vector_length(get_vector(l_2, p))

def quadratic_solver(b, C):
    # Para calcular as mascaras elipticas e cilindricas
    # b => centro da elipse (b_x)
    # C => (1 - sum(((k - b_k)/r_k)**2))*r_x**2
    if C < 0: 
        return False
    dx = sqrt(C)
    return [b - dx, b + dx]

def get_grid(L, N):
    return np.meshgrid(*[np.linspace(0, L[i], N[i]) for i in range(len(L))], indexing='ij')

def setup_plot(fig, L, N_mesh, scatter=False, p=1):
    dimensoes = len(L)
    if dimensoes == 2:
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        if not scatter:
            ax.set_xlim(-p, N_mesh[0] + p)
            ax.set_ylim(-p, N_mesh[1] + p)
    elif dimensoes == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect([1, 1, 1])
        if not scatter:
            ax.axes.set_xlim3d(-p, N_mesh[0] + p)
            ax.axes.set_ylim3d(-p, N_mesh[1] + p)
            ax.axes.set_zlim3d(-p, N_mesh[2] + p)
    if scatter:
        return ax, get_grid(L, N_mesh)
    return ax

def plot_vasculature(segments, nodes, widths, color='red', alpha=1):
    base_width = min(widths)
    relative_widths = [(w/base_width)**0.25 for w in widths]
    for i in range(len(segments)):
        plt.plot(*np.transpose([nodes[s] for s in segments[i]]), color=color, linewidth=relative_widths[i], alpha=alpha)


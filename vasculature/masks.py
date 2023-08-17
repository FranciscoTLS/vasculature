#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from .helpers import get_position, point_line_distance, quadratic_solver

def block_mask(shape, h, ranges):
    # ranges => [[start_x, end_x], [start_y, end_y], [start_z, end_z]]
    dimensions = len(ranges)
    index = [None]*(dimensions - 1)
    mask = np.ones(shape)
    
    def iterator(dimension=0):
        range_start = max(round(ranges[dimension][0]/h[dimension]), 0)
        range_end = min(round(ranges[dimension][1]/h[dimension]) + 1, shape[dimension])
        if dimension == dimensions - 1:
            for i in range(range_start, range_end):
                index_tuple = tuple(index + [i])
                mask[index_tuple] = 0
        else:
            for i in range(range_start, range_end):
                index[dimension] = i
                iterator(dimension + 1)
    
    iterator()
    return mask

def ellipsoid_mask(shape, h, ranges):
    # ranges => [[centre_x, R_x], [centre_y, R_y], [centre_z, R_z]]
    dimensions = len(ranges)
    index = [None]*(dimensions - 1)
    mask = np.ones(shape)
    
    def iterator(dimension=0, C=1):
        if dimension == dimensions - 1:
            solution = quadratic_solver(ranges[dimension][0], C*ranges[dimension][1]**2)
            if solution:
                range_start = max(round(solution[0]/h[dimension]), 0)
                range_end = min(round(solution[1]/h[dimension]) + 1, shape[dimension])
                for i in range(range_start, range_end):
                    index_tuple = tuple(index + [i])
                    mask[index_tuple] = 0
        else:
            range_start = max(round((ranges[dimension][0] - ranges[dimension][1])/h[dimension]), 0)
            range_end = min(round((ranges[dimension][0] + ranges[dimension][1])/h[dimension]) + 1, shape[dimension])
            for i in range(range_start, range_end):
                index[dimension] = i
                iterator(dimension + 1, C - ((i*h[dimension] - ranges[dimension][0])/ranges[dimension][1])**2)
                
    iterator()
    return mask

def cylinder_mask(shape, h, ranges):
    # ranges -> [[start_x, end_x], [centre_y, R_y], [centre_z, R_z]]
    dimensions = len(ranges)
    index = [None]*(dimensions - 1)
    mask = np.ones(shape)
    
    def iterator(dimension=0, C=1):
        if dimension == dimensions - 1:
            solution = quadratic_solver(ranges[dimension][0], C*ranges[dimension][1]**2)
            if solution:
                range_start = max(round(solution[0]/h[dimension]), 0)
                range_end = min(round(solution[1]/h[dimension]) + 1, shape[dimension])
                for i in range(range_start, range_end):
                    index_tuple = tuple(index + [i])
                    mask[index_tuple] = 0
        elif dimension == dimensions - 2:
            range_start = max(round((ranges[dimension][0] - ranges[dimension][1])/h[dimension]), 0)
            range_end = min(round((ranges[dimension][0] + ranges[dimension][1])/h[dimension]) + 1, shape[dimension])
            for i in range(range_start, range_end):
                index[dimension] = i
                iterator(dimension + 1, C - ((i*h[dimension] - ranges[dimension][0])/ranges[dimension][1])**2)
        else:
            range_start = max(round(ranges[dimension][0]/h[dimension]), 0)
            range_end = min(round(ranges[dimension][1]/h[dimension]) + 1, shape[dimension])
            for i in range(range_start, range_end):
                index[dimension] = i
                iterator(dimension + 1)
    
    iterator()
    return mask

def sausage_mask(shape, h, ranges, R):
    # ranges => [[x_1, x_2], [y_1, y_2], [z_1, z_2]]
    # R -> sausage radius
    dimensions = len(ranges)
    index = [None]*dimensions
    mask = np.ones(shape)
    
    def iterator(dimension=0):
        range_start = max(round((min(ranges[dimension]) - R)/h[dimension]), 0)
        range_end = min(round((max(ranges[dimension]) + R)/h[dimension]) + 1, shape[dimension])
        if dimension == dimensions - 1:
            for i in range(range_start, range_end):
                index[dimension] = i
                if point_line_distance(get_position(index, h), *np.transpose(ranges)) <= R:
                    index_tuple = tuple(index)
                    mask[index_tuple] = 0
        else:
            for i in range(range_start, range_end):
                index[dimension] = i
                iterator(dimension + 1)
                
    iterator()
    return mask


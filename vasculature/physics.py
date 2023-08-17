#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_radius(Fn, Ln, dP):
    # dP = abs(P_1 - P_2)
    # eta = 3.5e-3 -> (8*eta/pi)^(0.25) = 0.29564146
    return 0.29564146*(Fn*Ln/dP)**0.25

def update_radius(R_i, F_i, f):
    # Regra de três
    return R_i*(1 + f/F_i)**0.25


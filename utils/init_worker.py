# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:37:17 2025

@author: EH225
"""

# init_worker.py
import torch

def dask_setup(dask_worker):
    # set threads at worker startup
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

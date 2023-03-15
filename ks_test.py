#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:10:03 2023

@author: mikesha
"""
import torch

def alpha_D(D, n1: int, n2: int):
    return 2 * (-D.square() * 2 * n1 / (1 + n1 / n2)).exp()

@torch.jit.script
def kolmogorov_smirnov(points1, points2, alpha=torch.as_tensor([0.05, 0.01, 0.001, 0.0001])):
    """
    Kolmogorov-Smirnov test for empirical similarity of probability distributions.
    
    Warning: we assume that none of the elements of points1 coincide with points2. 
    The test may gave false negatives if there are coincidences, however the effect
    is small.

    Parameters
    ----------
    points1 : (..., n1) torch.Tensor
        Batched set of samples from the first distribution
    points2 : (..., n2) torch.Tensor
        Batched set of samples from the second distribution
    alpha : torch.Tensor
        Confidence intervals we wish to test. The default is torch.as_tensor([0.05, 0.01, 0.001, 0.0001]).

    Returns
    -------
    Tuple of (torch.Tensor, torch.Tensor)
        The test result at each alpha, and the estimated p-values.

    """
    n1 = points1.shape[-1]
    n2 = points2.shape[-1]
    
    # Confidence level
    c_ks = torch.sqrt(-0.5 * (alpha / 2).log())
    sup_conf = c_ks * torch.as_tensor((n1 + n2) / (n1 * n2)).sqrt()
    sup_conf = sup_conf.reshape((1, alpha.shape[0]))
    
    comb = torch.concatenate((points1, points2), dim=-1)
    
    comb_argsort = comb.argsort(dim=-1)
    
    pdf1 = torch.where(comb_argsort < n1, 1 / n1, 0)
    pdf2 = torch.where(comb_argsort >= n1, 1 / n2, 0)
    
    cdf1 = pdf1.cumsum(dim=-1)
    cdf2 = pdf2.cumsum(dim=-1)
    
    sup, _ = (cdf1 - cdf2).abs().max(dim=-1, keepdim=True)
    return sup > sup_conf, alpha_D(sup, n1 ,n2)

def test_uniform():
    p1 = torch.rand(1000)
    p2 = torch.rand(1000)
    print(kolmogorov_smirnov(p1, p2))
    assert not kolmogorov_smirnov(p1, p2)[0].all()

def test_norm_norm():
    p1 = torch.randn(1000)
    p2 = torch.randn(1000)
    print(kolmogorov_smirnov(p1, p2))
    assert not kolmogorov_smirnov(p1, p2)[0].all()
    
def test_unif_normal():
    p1 = torch.rand(1000)
    p2 = torch.randn(1000)
    print(kolmogorov_smirnov(p1, p2))
    assert kolmogorov_smirnov(p1, p2)[0].all()
    
def plot_sample_size_effect():
    batch = 100
    for n in [5, 10, 20, 30, 40, 50, 100, 1000]:
        s = (batch, n)
        normal = kolmogorov_smirnov(torch.randn(s), torch.randn(s))
        uniform = kolmogorov_smirnov(torch.rand(s), torch.rand(s))
        mixed = kolmogorov_smirnov(torch.rand(s), torch.randn(s))
        
        normal = normal[0].float().mean(dim=0), normal[1].mean()
        uniform = uniform[0].float().mean(dim=0), uniform[1].mean()
        mixed = mixed[0].float().mean(dim=0), mixed[1].mean()
        
        print(f'n: {n}, normal', normal[0], 'p:', normal[1])
        print(f'n: {n}, uniform', uniform[0], 'p:', uniform[1])
        print(f'n: {n}, mixed', mixed[0], 'p:', mixed[1])
        print()

"""LPIPS / FID / KID wrappers. No weight downloads: LPIPS runs with a random AlexNet (linear layers from the
package), FID/KID maths is checked on synthetic features with known answers."""

import numpy as np
import pytest
import torch

from src.evaluation.image_metrics import (InceptionFeatures, LPIPSMetric, MetricWeightsUnavailableError, frechet_distance,
                                          kernel_inception_distance)


def test_frechet_distance_known_values():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4000, 8))
    b = rng.normal(size=(4000, 8)) + 2.0  # mean shift of 2 in every dimension, same covariance
    assert frechet_distance(a, a) == pytest.approx(0, abs=1e-6)
    assert frechet_distance(a, b) == pytest.approx(8 * 2.0 ** 2, rel=0.05)  # ||mu1 - mu2||^2 = 32
    with pytest.raises(ValueError):
        frechet_distance(a[:1], b)


def test_kid_unbiased_and_sensitive():
    rng = np.random.default_rng(1)
    a, a2 = rng.normal(size=(800, 16)), rng.normal(size=(800, 16))
    b = rng.normal(size=(800, 16)) + 1.0
    same_mean, same_std = kernel_inception_distance(a, a2, n_subsets=20, subset_size=300)
    diff_mean, _ = kernel_inception_distance(a, b, n_subsets=20, subset_size=300)
    assert abs(same_mean) < 3 * same_std + 1e-3  # unbiased: ~0 for the same distribution
    assert diff_mean > 10 * max(abs(same_mean), 1e-4)


def test_lpips_identity_mask_and_gating(tmp_path):
    m = LPIPSMetric(torch.device("cpu"), random_backbone=True)
    a = torch.rand(2, 3, 64, 64)
    assert torch.allclose(m(a, a), torch.zeros(2), atol=1e-6)
    b = a.clone()
    b[:, :, :, 32:] = torch.rand(2, 3, 64, 32)  # change only the right half
    mask = torch.zeros(2, 64, 64, dtype=torch.bool)
    mask[:, :, :32] = True  # score only the left half
    assert torch.allclose(m(a, b, mask), torch.zeros(2), atol=1e-6)
    assert (m(a, b) > 0).all()
    with pytest.raises(MetricWeightsUnavailableError, match="allow_metric_download"):
        LPIPSMetric(torch.device("cpu"), cache_dir=tmp_path)


def test_inception_features_gating_and_injection(tmp_path):
    with pytest.raises(MetricWeightsUnavailableError, match="FID Inception"):
        InceptionFeatures(torch.device("cpu"), cache_dir=tmp_path)
    f = InceptionFeatures(torch.device("cpu"), extractor=lambda x: x.mean((2, 3)))
    out = f(torch.rand(5, 3, 16, 16))
    assert out.shape == (5, 3) and out.dtype == np.float64

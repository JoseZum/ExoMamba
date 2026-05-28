"""Pruebas unitarias para src/exoplanet/data/augment.py.

Verifican:
 - Shapes y dtypes preservados.
 - No mutación del input (devolución de tensor nuevo).
 - Reproducibilidad bajo torch.Generator.
 - Rango razonable de la salida (no se sale de lo esperado).
 - Compose aplica las transformaciones en orden.
 - build_augment_pipeline construye Compose desde dict de YAML.
"""

from __future__ import annotations

import pytest
import torch

from exoplanet.data.augment import (
    Compose,
    amplitude_scale,
    build_augment_pipeline,
    gaussian_noise,
    temporal_shift,
    time_reverse,
)


def _make_curve(length: int = 18000, channels: int = 1) -> torch.Tensor:
    """Curva sintética estilo flujo normalizado (~1.0 con micro-variación)."""
    torch.manual_seed(0)
    base = torch.ones((channels, length), dtype=torch.float32)
    base += torch.randn_like(base) * 0.001
    return base


# ---------- temporal_shift ----------


def test_temporal_shift_preserva_shape_y_dtype() -> None:
    x = _make_curve()
    out = temporal_shift(x, max_shift=500)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_temporal_shift_no_muta_input() -> None:
    x = _make_curve()
    snapshot = x.clone()
    _ = temporal_shift(x, max_shift=500)
    assert torch.equal(x, snapshot)


def test_temporal_shift_reproducible_con_generator() -> None:
    x = _make_curve()
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = temporal_shift(x, max_shift=500, generator=g1)
    b = temporal_shift(x, max_shift=500, generator=g2)
    assert torch.equal(a, b)


def test_temporal_shift_max_shift_cero_es_clon() -> None:
    x = _make_curve()
    out = temporal_shift(x, max_shift=0)
    assert torch.equal(out, x)
    assert out is not x


def test_temporal_shift_1d_tensor() -> None:
    x = _make_curve().squeeze(0)  # (L,)
    out = temporal_shift(x, max_shift=500)
    assert out.shape == x.shape


# ---------- gaussian_noise ----------


def test_gaussian_noise_preserva_shape_y_dtype() -> None:
    x = _make_curve()
    out = gaussian_noise(x, sigma=0.001)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gaussian_noise_no_muta_input() -> None:
    x = _make_curve()
    snapshot = x.clone()
    _ = gaussian_noise(x, sigma=0.01)
    assert torch.equal(x, snapshot)


def test_gaussian_noise_rango_razonable() -> None:
    x = _make_curve()
    g = torch.Generator().manual_seed(1)
    out = gaussian_noise(x, sigma=0.001, generator=g)
    # Diferencia esperada: la mayoría dentro de ±5*sigma (~0.005)
    diff = (out - x).abs()
    assert diff.max().item() < 0.01  # 10*sigma como cota laxa


def test_gaussian_noise_reproducible() -> None:
    x = _make_curve()
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    a = gaussian_noise(x, sigma=0.001, generator=g1)
    b = gaussian_noise(x, sigma=0.001, generator=g2)
    assert torch.equal(a, b)


def test_gaussian_noise_sigma_cero_es_clon() -> None:
    x = _make_curve()
    out = gaussian_noise(x, sigma=0.0)
    assert torch.equal(out, x)
    assert out is not x


# ---------- time_reverse ----------


def test_time_reverse_prob_uno_invierte() -> None:
    x = _make_curve()
    out = time_reverse(x, prob=1.0)
    assert torch.equal(out, torch.flip(x, dims=[-1]))


def test_time_reverse_prob_cero_no_invierte() -> None:
    x = _make_curve()
    out = time_reverse(x, prob=0.0)
    assert torch.equal(out, x)
    assert out is not x


def test_time_reverse_no_muta_input() -> None:
    x = _make_curve()
    snapshot = x.clone()
    _ = time_reverse(x, prob=1.0)
    assert torch.equal(x, snapshot)


def test_time_reverse_reproducible() -> None:
    x = _make_curve()
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    a = time_reverse(x, prob=0.5, generator=g1)
    b = time_reverse(x, prob=0.5, generator=g2)
    assert torch.equal(a, b)


# ---------- amplitude_scale ----------


def test_amplitude_scale_preserva_shape_y_dtype() -> None:
    x = _make_curve()
    out = amplitude_scale(x, low=0.99, high=1.01)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_amplitude_scale_no_muta_input() -> None:
    x = _make_curve()
    snapshot = x.clone()
    _ = amplitude_scale(x, low=0.99, high=1.01)
    assert torch.equal(x, snapshot)


def test_amplitude_scale_factor_en_rango() -> None:
    x = _make_curve()
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        out = amplitude_scale(x, low=0.99, high=1.01, generator=g)
        # ratio puntual estable porque la op es x * factor escalar
        ratios = (out / x).flatten()
        # Tomamos la mediana para evadir outliers numéricos
        ratio = ratios.median().item()
        assert 0.99 - 1e-5 <= ratio <= 1.01 + 1e-5


def test_amplitude_scale_low_igual_high() -> None:
    x = _make_curve()
    out = amplitude_scale(x, low=1.0, high=1.0)
    assert torch.allclose(out, x)


def test_amplitude_scale_low_mayor_que_high_lanza_error() -> None:
    x = _make_curve()
    with pytest.raises(ValueError):
        amplitude_scale(x, low=1.5, high=0.5)


# ---------- Compose y registry ----------


def test_compose_aplica_en_orden() -> None:
    x = _make_curve()
    g = torch.Generator().manual_seed(42)
    aug = Compose([
        lambda t, generator=None: gaussian_noise(t, sigma=0.001, generator=generator),
        lambda t, generator=None: amplitude_scale(t, low=1.0, high=1.0, generator=generator),
    ])
    out = aug(x, generator=g)
    # Como amplitude_scale es identidad (low=high=1.0), out debe igual a gaussian-noise(x)
    g2 = torch.Generator().manual_seed(42)
    expected = gaussian_noise(x, sigma=0.001, generator=g2)
    assert torch.equal(out, expected)


def test_compose_reproducible() -> None:
    x = _make_curve()
    pipeline = [
        {"type": "temporal_shift", "max_shift": 500},
        {"type": "gaussian_noise", "sigma": 0.001},
        {"type": "amplitude_scale", "low": 0.99, "high": 1.01},
    ]
    aug1 = build_augment_pipeline(pipeline)
    aug2 = build_augment_pipeline(pipeline)
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    a = aug1(x, generator=g1)
    b = aug2(x, generator=g2)
    assert torch.equal(a, b)


def test_build_augment_pipeline_tipo_desconocido() -> None:
    with pytest.raises(ValueError, match="desconocida"):
        build_augment_pipeline([{"type": "no_existe"}])


def test_build_augment_pipeline_entrada_invalida() -> None:
    with pytest.raises(ValueError, match="inválida"):
        build_augment_pipeline([{"sin_type": True}])


def test_compose_vacio() -> None:
    x = _make_curve()
    aug = Compose([])
    out = aug(x)
    assert torch.equal(out, x)

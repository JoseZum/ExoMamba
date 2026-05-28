"""
Augmentations para curvas de luz (Fase 8 — sweep Tier 1).

Todas las funciones operan sobre tensores `(L,)` o `(L, 1)` (también soportan
`(1, L)`, que es la forma usada por LightCurveDataset). NO mutan el input:
siempre devuelven un tensor nuevo. Aceptan un `torch.Generator` opcional para
reproducibilidad de un sample dado.

Restricción operativa crítica (de CLAUDE.md):
    El augmentation se aplica SOLO en train. El caller (Dataset / runner) es
    responsable de no pasar augmentations a val ni test. Las funciones de este
    módulo no saben en qué split están corriendo.

Diseño del Compose: equivalente liviano a torchvision.transforms.Compose, pero
sin imponer torchvision como dependencia adicional. Cada augmentation recibe el
mismo `generator` para que toda la pipeline de un sample sea reproducible.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch


def _rand(generator: torch.Generator | None) -> torch.Tensor:
    """Devuelve un escalar uniforme en [0, 1) en CPU usando el generator dado."""
    return torch.rand(1, generator=generator)


def temporal_shift(
    x: torch.Tensor,
    max_shift: int = 500,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Desplaza la curva ±max_shift puntos rellenando con la mediana del propio sample.

    El shift `s` se muestrea uniformemente en `[-max_shift, max_shift]`. Para
    `s > 0` la curva se mueve a la derecha (los primeros `s` puntos se rellenan
    con la mediana); para `s < 0` la curva se mueve a la izquierda (los últimos
    `|s|` puntos se rellenan con la mediana).

    Decisión de boundary: rellenamos con la mediana del propio sample (no
    wrap-around). Wrap-around inyectaría un salto artificial entre el final y
    el inicio que el modelo podría aprender como feature espuria. La mediana es
    el flujo "típico" fuera de tránsito y es lo que `preprocess_global.py` ya
    usa como nivel base post-normalización.

    Args:
        x: tensor `(L,)`, `(L, 1)` o `(1, L)` con la curva.
        max_shift: shift máximo absoluto en puntos. Si <= 0 retorna `x.clone()`.
        generator: torch.Generator opcional para reproducibilidad.

    Returns:
        Tensor nuevo con la misma forma que `x`.
    """
    if max_shift <= 0:
        return x.clone()

    # Muestreamos shift entero en [-max_shift, max_shift]
    rand_val = _rand(generator).item()
    shift = int(round((rand_val * 2.0 - 1.0) * max_shift))
    if shift == 0:
        return x.clone()

    out = x.clone()
    fill_value = float(x.median().item())

    # Detectamos el eje temporal: la convención del repo es `(1, L)` o `(L,)`.
    if out.dim() == 1:
        L = out.shape[0]
        if abs(shift) >= L:
            out.fill_(fill_value)
            return out
        if shift > 0:
            out[shift:] = x[:-shift]
            out[:shift] = fill_value
        else:
            s = -shift
            out[:-s] = x[s:]
            out[-s:] = fill_value
        return out

    # Tensor 2D: asumimos que la dim de mayor tamaño es el tiempo.
    time_dim = int(torch.tensor(out.shape).argmax().item())
    L = out.shape[time_dim]
    if abs(shift) >= L:
        out.fill_(fill_value)
        return out

    # Usamos torch.roll sobre time_dim y luego machacamos el "wrap" con fill_value.
    out = torch.roll(x, shifts=shift, dims=time_dim).clone()
    if shift > 0:
        idx = torch.arange(shift)
    else:
        idx = torch.arange(L + shift, L)
    out.index_fill_(time_dim, idx, fill_value)
    return out


def gaussian_noise(
    x: torch.Tensor,
    sigma: float = 0.001,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Suma ruido N(0, sigma) en la escala del flujo normalizado.

    Sigma default = 0.001 ≈ ruido instrumental típico tras PDCSAP_FLUX
    normalizado por mediana (el flujo vive en ~[0.99, 1.01], dips de tránsito
    son 0.01% – 1%). Sigma demasiado alto destruiría la señal de tránsito.

    Args:
        x: tensor `(L,)`, `(L, 1)` o `(1, L)`.
        sigma: desviación estándar del ruido aditivo.
        generator: torch.Generator opcional para reproducibilidad.

    Returns:
        Tensor nuevo con la misma forma y dtype que `x`.
    """
    if sigma <= 0:
        return x.clone()
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype) * sigma
    return x + noise


def time_reverse(
    x: torch.Tensor,
    prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Invierte temporalmente la curva con probabilidad `prob`.

    Los tránsitos de exoplanetas son aproximadamente simétricos (igual ingreso
    y egreso), por lo que invertir el eje del tiempo es una augmentation válida
    para Tier 1 (vista global pura). NO aplicar en Tier 2 sobre la vista local
    phase-folded si esa se llegara a augmentar.

    Args:
        x: tensor `(L,)`, `(L, 1)` o `(1, L)`.
        prob: probabilidad de aplicar la inversión. 0 = nunca, 1 = siempre.
        generator: torch.Generator opcional para reproducibilidad.

    Returns:
        Tensor nuevo con la misma forma que `x`. Si no se invierte, es un
        `x.clone()` (nunca el mismo objeto del input).
    """
    if prob <= 0:
        return x.clone()
    if _rand(generator).item() >= prob:
        return x.clone()

    if x.dim() == 1:
        return torch.flip(x, dims=[0])
    time_dim = int(torch.tensor(x.shape).argmax().item())
    return torch.flip(x, dims=[time_dim])


def amplitude_scale(
    x: torch.Tensor,
    low: float = 0.99,
    high: float = 1.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Multiplica todo el flujo por un factor uniforme en [low, high].

    Simula pequeñas diferencias de calibración entre sectores TESS sin alterar
    la forma del tránsito (profundidad y duración escalan proporcionalmente).
    El rango default [0.99, 1.01] es consistente con la propuesta original.

    Args:
        x: tensor `(L,)`, `(L, 1)` o `(1, L)`.
        low: factor mínimo.
        high: factor máximo.
        generator: torch.Generator opcional para reproducibilidad.

    Returns:
        Tensor nuevo con la misma forma y dtype que `x`.
    """
    if low > high:
        raise ValueError(f"low ({low}) > high ({high})")
    if low == high:
        return x * low
    factor = low + (_rand(generator).item() * (high - low))
    return x * factor


class Compose:
    """Aplica una secuencia de augmentations en orden.

    Equivalente liviano a torchvision.transforms.Compose. Cada augmentation
    recibe el mismo `torch.Generator` para que toda la pipeline de un sample
    sea reproducible.

    Ejemplo:
        >>> aug = Compose([
        ...     lambda x, generator=None: temporal_shift(x, 500, generator=generator),
        ...     lambda x, generator=None: gaussian_noise(x, 0.001, generator=generator),
        ... ])
        >>> y = aug(x)
    """

    def __init__(self, transforms: Sequence[Callable[..., torch.Tensor]]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        out = x
        for t in self.transforms:
            out = t(out, generator=generator)
        return out

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        names = [getattr(t, "__name__", repr(t)) for t in self.transforms]
        return f"Compose([{', '.join(names)}])"


# Registry de augmentations por nombre — usado por runner.py al construir el
# pipeline desde el YAML. Cada entrada es una factory que recibe el dict de
# kwargs del YAML y devuelve una función `(x, generator=None) -> tensor`.
def _build_temporal_shift(params: dict) -> Callable[..., torch.Tensor]:
    max_shift = int(params.get("max_shift", 500))

    def _fn(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return temporal_shift(x, max_shift=max_shift, generator=generator)

    _fn.__name__ = f"temporal_shift(max_shift={max_shift})"
    return _fn


def _build_gaussian_noise(params: dict) -> Callable[..., torch.Tensor]:
    sigma = float(params.get("sigma", 0.001))

    def _fn(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return gaussian_noise(x, sigma=sigma, generator=generator)

    _fn.__name__ = f"gaussian_noise(sigma={sigma})"
    return _fn


def _build_time_reverse(params: dict) -> Callable[..., torch.Tensor]:
    prob = float(params.get("prob", 0.5))

    def _fn(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return time_reverse(x, prob=prob, generator=generator)

    _fn.__name__ = f"time_reverse(prob={prob})"
    return _fn


def _build_amplitude_scale(params: dict) -> Callable[..., torch.Tensor]:
    low = float(params.get("low", 0.99))
    high = float(params.get("high", 1.01))

    def _fn(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return amplitude_scale(x, low=low, high=high, generator=generator)

    _fn.__name__ = f"amplitude_scale(low={low}, high={high})"
    return _fn


AUGMENT_REGISTRY: dict[str, Callable[[dict], Callable[..., torch.Tensor]]] = {
    "temporal_shift": _build_temporal_shift,
    "gaussian_noise": _build_gaussian_noise,
    "time_reverse": _build_time_reverse,
    "amplitude_scale": _build_amplitude_scale,
}


def build_augment_pipeline(spec: list[dict]) -> Compose:
    """Construye un Compose desde una lista de dicts del YAML.

    Cada dict debe tener al menos la clave `type` (string) y opcionalmente los
    parámetros específicos del augmentation. Ejemplo:

        spec = [
            {"type": "temporal_shift", "max_shift": 500},
            {"type": "gaussian_noise", "sigma": 0.001},
        ]
    """
    transforms = []
    for entry in spec:
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(f"Entrada inválida en pipeline de augmentation: {entry}")
        kind = entry["type"]
        if kind not in AUGMENT_REGISTRY:
            raise ValueError(
                f"Augmentation desconocida: '{kind}'. "
                f"Disponibles: {sorted(AUGMENT_REGISTRY.keys())}"
            )
        params = {k: v for k, v in entry.items() if k != "type"}
        transforms.append(AUGMENT_REGISTRY[kind](params))
    return Compose(transforms)


__all__ = [
    "temporal_shift",
    "gaussian_noise",
    "time_reverse",
    "amplitude_scale",
    "Compose",
    "build_augment_pipeline",
    "AUGMENT_REGISTRY",
]

# Setup de WSL2 + Ubuntu 24.04 para entrenar Mamba (Fase 8)

> **Por qué hace falta esto:** `mamba-ssm` no tiene wheels precompilados para
> Windows nativo. Hay que compilarlo desde fuente, lo cual requiere `nvcc`
> (compilador CUDA) que en la práctica solo funciona limpio en Linux.
> WSL2 nos da Linux dentro de Windows sin tener que dual-bootear.
>
> Esta guía es de cero a Mamba entrenando. Es **una vez**, después solo
> abrís Ubuntu y `source .venv/bin/activate` y listo.

---

## Lo que ya está hecho (no toques)

El proyecto en Windows tiene todo el pipeline corriendo (datos descargados,
preprocesados, splits, CNN entrenada). Los `.pt` y splits viven en
`data/processed/global/` y `data/splits/` y son **compatibles con Linux
sin conversión** (formato PyTorch nativo). En WSL2 vas a usar los mismos
archivos via el bridge de filesystem.

---

## Pre-requisitos (en Windows host)

1. **Windows 11** (ya cumplido — confirmado en CLAUDE.md).
2. **Driver NVIDIA actualizado** (≥ 525 para CUDA 12.x).
   Verificar abriendo PowerShell:

   ```powershell
   nvidia-smi.exe
   ```

   Deberías ver tu RTX 3050 y el "Driver Version" en la esquina superior derecha.
3. **Espacio en disco**: ~15 GB libres en C: (Ubuntu + dependencias + venv).

---

## Paso 1 — Instalar WSL2 + Ubuntu 24.04

Abrí **PowerShell como administrador** y corré:

```powershell
wsl --install -d Ubuntu-24.04
```

Esto instala:
- Características WSL2 de Windows (si faltan).
- Distribución Ubuntu 24.04 desde Microsoft Store.

**Probable: te pide reiniciar Windows.** Hacelo.

Después del reinicio, Ubuntu se inicia solo y te pide:
- Un username (cualquiera, ej. `joche`).
- Una contraseña (cualquiera, fácil de recordar — la vas a usar para `sudo`).

Cuando termine vas a ver el prompt Linux:

```
joche@DESKTOP-XXX:~$
```

**Verificación:** desde PowerShell host:

```powershell
wsl -l -v
```

Deberías ver `Ubuntu-24.04` con `STATE = Running` y `VERSION = 2`.

---

## Paso 2 — Acceder al proyecto desde Ubuntu

El filesystem de Windows está montado en `/mnt/c/`. Desde Ubuntu:

```bash
cd /mnt/c/Users/jfzum/Downloads/Proyecto-IA/mamba-exoplanet
ls
```

Deberías ver `BITACORA.md`, `pyproject.toml`, `data/`, etc.

> **Performance:** trabajar desde `/mnt/c/` es ~10x más lento que trabajar
> en el filesystem nativo de WSL2 (`~/`). Para entrenamiento real puede
> importar. Si querés, clonás el repo dentro de `~/` y trabajás desde ahí,
> pero perderías la sincronización automática con Windows. Para este proyecto,
> usar `/mnt/c/` está bien — la cuello de botella va a ser la GPU, no el disco.

---

## Paso 3 — Correr el script de setup

Una sola línea:

```bash
chmod +x scripts/setup_wsl2.sh
./scripts/setup_wsl2.sh
```

Esto hace **todo** automáticamente:

1. `sudo apt update` y dependencias del sistema (gcc, python3.11, nvidia-cuda-toolkit).
2. Crea `.venv` con Python 3.11.
3. Instala PyTorch CUDA 12.x.
4. Instala el paquete `exoplanet` con `[dev]` y `[mamba]`.
5. Corre `verify_wsl2_env.py` que chequea que todo quedó bien.

**Va a pedir tu password de sudo varias veces** (para apt).

**Va a tardar ~15-25 min** la primera vez. La compilación de `mamba-ssm`
es lo más lento (5-15 min en sí mismo).

Si todo sale bien, al final ves:

```
[OK] TODO PASA. El entorno WSL2 está listo para entrenar Mamba.
```

---

## Paso 4 — Verificar y correr smoke train

Activá el venv y corré el preflight con tensores random:

```bash
source .venv/bin/activate
python scripts/smoke_train_mamba.py
```

Esperado: 3 pasos de entrenamiento sobre tensores fake `(16, 18000, 1)`,
sin OOM, sin errores. Tarda ~30 s.

---

## Paso 5 — Entrenamiento real de Mamba

```bash
python scripts/train.py --config configs/mamba_small.yaml
```

Tiempo estimado: 1-2 h en RTX 3050. Va a usar los `.pt` que ya
preprocessamos desde Windows (compatible).

Mientras corre, en otra terminal de Ubuntu podés abrir TensorBoard:

```bash
source .venv/bin/activate
tensorboard --logdir experiments/
```

Y abrís http://localhost:6006 en el navegador (Windows o Linux, da igual).

---

## Troubleshooting

### `nvidia-smi` no aparece en WSL2

- Asegurate que el driver NVIDIA está en Windows host (no necesitás
  instalar nada de NVIDIA dentro de WSL2 — el driver de Windows expone
  la GPU automáticamente vía WSL).
- En PowerShell host: `wsl --shutdown` y volvé a abrir Ubuntu.
- Si sigue: actualizá driver NVIDIA en Windows desde GeForce Experience.

### `nvcc: command not found`

```bash
sudo apt install nvidia-cuda-toolkit
```

### `pip install causal-conv1d` falla con `nvcc fatal error`

Versión incompatible de CUDA. Verificar:

```bash
nvcc --version              # CUDA del toolkit
python -c "import torch; print(torch.version.cuda)"   # CUDA del torch
```

Tienen que ser **major version compatibles** (ej. CUDA 12.x ↔ torch cu12x).
Si no, reinstalar torch con la versión correcta:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(o `cu118` si tu nvcc es 11.x).

### `pip install mamba-ssm` se queda colgado

Es normal — está compilando código CUDA. Puede tardar 10 min sin
imprimir nada. Esperá. Si pasa de 30 min sin terminar, cancelá con
`Ctrl+C` y probá con flag de paciencia:

```bash
pip install --no-build-isolation -v mamba-ssm
```

El `-v` muestra el progreso de compilación.

### OOM (out of memory) al entrenar

Editá `configs/mamba_small.yaml`:
- Bajar `model.params.d_model` de 64 a 32.
- O activar `training.gradient_checkpointing: true`.
- O bajar `data.batch_size` de 16 a 8.

### El entrenamiento es lentísimo (> 5 min por epoch)

- Estás corriendo desde `/mnt/c/`? Ese filesystem es lento desde WSL2.
  Probá clonar el repo a `~/mamba-exoplanet` y trabajar de ahí.
- Asegurate que `data.num_workers > 0` en el YAML (ej. 2 o 4).
- Verificá que la GPU está siendo usada: `nvidia-smi` mientras
  entrena debería mostrar `python` consumiendo VRAM.

---

## Resumen visual

```
Windows host
  └── Driver NVIDIA   ← (ya instalado)
  └── PowerShell admin
       └── wsl --install -d Ubuntu-24.04   ← Paso 1

WSL2 Ubuntu 24.04
  └── /mnt/c/Users/jfzum/Downloads/Proyecto-IA/mamba-exoplanet
       ├── ./scripts/setup_wsl2.sh         ← Paso 3 (una vez)
       │    ├── apt deps + nvcc
       │    ├── python3.11 + venv
       │    ├── pip install torch + cu121
       │    ├── pip install -e ".[dev,mamba]"
       │    └── verify_wsl2_env.py         ← chequea todo
       │
       ├── ./scripts/smoke_train_mamba.py  ← Paso 4 (preflight)
       └── ./scripts/train.py --config configs/mamba_small.yaml   ← Paso 5
```

---

## Después de Mamba

Una vez que tengas el `best.pt` de Mamba en `experiments/<timestamp>_mamba_small/`,
Fase 9 (evaluación final Tier 1) va a:

1. Cargar `best.pt` de CNN y de Mamba.
2. Evaluar ambos sobre `test_tics.csv` (sellado desde Fase 4).
3. Generar tabla comparativa + curvas ROC/PR + análisis de errores.
4. Ese resultado va al paper.

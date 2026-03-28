# Nanobot Chemotactic Plume Tracking
### Robothon 2026 — Problem Statement 4-b | GLA University, Mathura | Team: Squirtle Squad

> Autonomous 3D navigation of a nanorobot toward a chemical source using gradient climbing and Kalman filtering — without any map, using only local concentration sensing.

---

## Team — Squirtle Squad

| Member | Role |
|--------|------|
| **Yash Yaduvanshi** (Team Lead) | Algorithm design, Kalman filter implementation, simulation architecture |
| **Sujal Yadav** | Visualization, animation system, interactive 3D viewer |
| **Anoop Kumar** | Testing, edge case validation, performance analysis |
| **Bhavishya Goyal** | Documentation, presentation, real-world application research |

- GitHub: [github.com/yash2yaduvanshi/Robothon-2026](https://github.com/yash2yaduvanshi/Robothon-2026)

---

## Problem Statement (4-b)
Simulate a nanorobot finding a chemical source in a **3D space** by following a chemical trail.  
The source releases a biomarker following a **Gaussian distribution** — concentration gets weaker with distance.  
The nanorobot only knows the concentration at its current position and must find the source in the **shortest time possible** using a gradient-climbing algorithm.

**Constraint:** Space is a bounded 10×10×10 cube. Any coordinate outside [0, 10] is invalid.

---

## How It Works

### Chemical Environment
The source emits a biomarker with 3D Gaussian concentration decay:

```
C(x, y, z) = C_peak × exp(−d² / 2σ²)
```

Where `d` is Euclidean distance from source and `σ` controls plume spread.

### Algorithm: Gradient Climbing + Kalman Filter

1. **Sensor reading** — nanobot reads concentration at current position (±3% Gaussian noise)
2. **Kalman filter** — cleans the noisy reading using optimal 1D state estimation
3. **Gradient estimation** — probes 6 neighbours (±x, ±y, ±z) using central finite differences
4. **Navigation** — moves in direction of steepest concentration increase
5. **Fallback** — random walk when gradient is too weak (flat zone escape)
6. **Goal check** — stops when within `goal_radius` of source

### Kalman Filter
Maintains estimate `x` (concentration) and uncertainty `P`:
```
Predict:  P = P + Q
Gain:     K = P / (P + R)
Update:   x = x + K × (z − x),   P = (1 − K) × P
```
- **Q = 5.0** — process noise (derived from gradient × step_size)
- **R = 900.0** — measurement noise (= (0.03 × 1000)² = 900, physically derived)
- **K ≈ 0.07** — steady-state gain (7% sensor trust, 93% model trust)

---

## Files
```
Robothon-2026/
├── nanobot_dynamic.py      # Main simulation → saves animated GIF
├── nanobot_view3d.py       # Interactive live viewer (drag-to-rotate 3D)
├── metrics.json            # Performance data (auto-generated)
├── nanobot_dynamic.gif     # Animated output (auto-generated)
├── presentation.pptx       # Technical presentation
└── README.md               # This file
```

---

## Installation

### Requirements
- Python 3.8+
- pip

### Step 1 — Clone
```bash
git clone https://github.com/yash2yaduvanshi/Robothon-2026.git
cd Robothon-2026
```

### Step 2 — Install dependencies
```bash
pip install numpy matplotlib pillow
```

### Step 3 — Run simulation (saves GIF)
```bash
python nanobot_dynamic.py
```

### Step 4 — Run interactive viewer
```bash
pip install tk
python nanobot_view3d.py
```

---

## Configuration
Edit `CONFIG` at the top of either file:

```python
CONFIG = {
    "source_pos"   : [9.0, 6.0, 2.0],   # Chemical source — must be in [0, 10]
    "start_pos"    : [0.4, 0.5, 0.2],   # Nanobot start — negative = clamped to wall
    "space_size"   : 10.0,              # Cube size — fixed, do not change
    "sigma"        : 3.5,               # Plume spread (2.0–6.0 recommended)
    "peak_conc"    : 1000.0,            # Max concentration at source
    "step_size"    : 0.25,              # Movement per step (keep < goal_radius × 2)
    "noise_level"  : 0.03,             # Sensor noise = 3%
    "max_steps"    : 400,              # Max iterations
    "goal_radius"  : 0.35,             # Success threshold (keep > step_size / 2)
    "sample_delta" : 0.3,              # Gradient probe distance
    "kf_Q"         : 5.0,              # Kalman process noise
    "kf_R"         : 900.0,            # Kalman measurement noise
}
```

### Edge Cases Handled
| Input | Behaviour |
|-------|-----------|
| Source outside [0,10] | WARNING + False + static error frame |
| Start negative | WARNING + clamped to nearest wall |
| step_size too small | Runs out of steps → False |
| goal_radius = 0 | Mathematically unreachable → False |
| max_steps = 0 | Instant False |
| Source = Start | Instant Found: True |

---

## Output — 6 Live Panels
| Panel | Shows |
|-------|-------|
| 3D Navigation Path | Nanobot trajectory, drag to rotate |
| Kalman: Raw vs Filtered | Gold = noisy, Orange = Kalman cleaned |
| Distance to Source | Drops to zero at goal |
| Gradient Strength | How strongly field points toward source |
| Kalman Gain K | Filter convergence to steady state ~0.07 |
| Live Metrics | Step, distance, efficiency, mode, K value |

---

## Performance (averaged over 20 runs)
| Metric | Value |
|--------|-------|
| Steps to source | ~55 |
| Path efficiency | ~78% |
| Final distance | < 0.35 u |
| Kalman gain K | 0.07 (steady state) |

---

## Real-World Application
This models how **medical nanobots** could:
- Navigate bloodstream to locate cancer cells via tumor biomarkers (PSA, CEA)
- Deliver targeted drugs directly at the tumor site
- Track infection via inflammatory cytokines (cytokine-directed navigation)
- Operate without GPS or external guidance

---

## License
MIT License — free to use and modify.

"""
=============================================================
  Robothon 2026 - Project 4b: Chemotactic Plume Tracking
  ROBOTiX Club, NIT Raipur | ANANTYA'26
=============================================================
  Author  : Yash Yaduvanshi
  GitHub  : https://github.com/yash2yaduvanshi
  LinkedIn: https://www.linkedin.com/in/yash-yaduvanshi-2f2008/

  DYNAMIC ANIMATED VERSION with Kalman Filter
  All 6 panels update live — nanobot moves in real time

  FIXES over v1:
    - KalmanFilter1D initialised with first sensor reading (not 0)
    - Direction normalisation guarded against zero-vector crash
    - Filtered concentration clamped to >= 0
    - Efficiency precomputed per-frame — no O(n^2) recomputation
    - 3D panes set transparent (clean dark look)
    - 3D grid and axis-plane colors properly styled
    - Removed unused 'import time'
    - Animation interval tuned for smooth GIF output
    - Camera rotation step halved for smoother spin
    - All panels sized / spaced cleanly
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless render — remove for interactive window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
import json

# ─── SIMULATION PARAMETERS ───────────────────────────────────────────────────
CONFIG = {
    "source_pos"   : [9.0, 6.0, 2.0],   # chemical source location
    "start_pos"    : [0.4, 0.5, 0.2],   # nanobot start position
    "space_size"   : 10.0,              # 10x10x10 bounded space (units)
    "sigma"        : 3.5,               # Gaussian spread of chemical plume
    "peak_conc"    : 1000.0,            # maximum concentration at source
    "step_size"    : 0.25,              # movement distance per step
    "noise_level"  : 0.03,             # sensor noise = 3% of true reading
    "max_steps"    : 400,              # upper bound on navigation steps
    "goal_radius"  : 0.35,             # success threshold (units from source)
    "sample_delta" : 0.3,              # finite-difference probe distance
    # Kalman filter
    "kf_Q"         : 5.0,              # process noise variance
    "kf_R"         : 900.0,            # measurement noise variance
    # Rule of thumb: noise_level=0.03, peak=1000 -> std~30 -> R~900
}

SOURCE_POS = np.array(CONFIG["source_pos"])
START_POS  = np.array(CONFIG["start_pos"])

# ── Space is always fixed 10x10x10 ──────────────────────────────────────────
SPACE = CONFIG["space_size"]   # = 10.0, never changes

def _in_bounds(pos):
    """Return True if all coordinates are within [0, SPACE]."""
    return bool(np.all(np.array(pos) >= 0.0) and np.all(np.array(pos) <= SPACE))

# NOTE: SOURCE_IN_BOUNDS and START_IN_BOUNDS are computed inside run_simulation()
# and build_and_save() using the live CONFIG values — not here at module load time.
# This ensures user edits to CONFIG are picked up correctly.

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
BG_C   = "#0a0a1a"   # figure background (near black)
PANEL  = "#111133"   # subplot face colour
CYAN   = "#00e5ff"   # path / distance line
PINK   = "#ff4081"   # source marker / goal line
GREEN  = "#69ff47"   # nanobot dot / gradient mode
GOLD   = "#ffd740"   # raw concentration
WHITE  = "#ffffff"
GRAY   = "#aaaacc"
PURP   = "#b967ff"   # gradient strength line
ORANGE = "#ff9100"   # Kalman-filtered concentration


# ─── KALMAN FILTER ────────────────────────────────────────────────────────────
class KalmanFilter1D:
    """
    1-D Kalman filter -- fuses a noisy scalar sensor with a prediction model.

    STATE  x : estimated true concentration (scalar)
    COVAR  P : uncertainty in x (variance)

    Every call to step(z):
        PREDICT  ->  P = P + Q           (uncertainty grows as nanobot moves)
        GAIN     ->  K = P / (P + R)     (how much to trust new sensor reading)
        UPDATE   ->  x = x + K*(z - x)  (blend prediction with measurement)
                     P = (1 - K) * P    (uncertainty shrinks after update)

    Parameters
    ----------
    init_val : float   Initial estimate for x. Use first noise-free reading.
    Q        : float   Process noise variance. How fast does true conc change?
    R        : float   Measurement noise variance. How noisy is the sensor?
    """
    def __init__(self, init_val: float = 0.0, Q: float = 5.0, R: float = 900.0):
        self.x = float(init_val)   # state estimate
        self.P = float(R)          # start uncertain -- set P = R initially
        self.Q = float(Q)
        self.R = float(R)

    def step(self, z: float):
        """
        One Kalman predict-update cycle.

        Parameters
        ----------
        z : float   Noisy measurement from sensor.

        Returns
        -------
        x_est : float   Filtered (smoothed) estimate of true value.
        K     : float   Kalman gain (0 -> trust model, 1 -> trust sensor).
        """
        # PREDICT: uncertainty grows
        self.P += self.Q

        # KALMAN GAIN: how much weight to give the sensor
        K = self.P / (self.P + self.R)    # always in (0, 1)

        # UPDATE: blend prediction with measurement
        self.x = self.x + K * (z - self.x)  # innovation = (z - x)
        self.P = (1.0 - K) * self.P          # variance shrinks

        return self.x, K


# ─── CHEMICAL CONCENTRATION FIELD ────────────────────────────────────────────
def concentration(pos, noise: bool = True) -> float:
    """
    3-D Gaussian chemical plume centred at SOURCE_POS.

        C(r) = C_peak * exp( -||r - r_src||^2 / (2*sigma^2) )

    With noise=True, adds Gaussian noise of std = noise_level * C.
    Returns value clamped to [0, inf).
    """
    dist_sq = float(np.sum((np.asarray(pos) - SOURCE_POS) ** 2))
    c = CONFIG["peak_conc"] * np.exp(-dist_sq / (2.0 * CONFIG["sigma"] ** 2))
    if noise:
        c += np.random.normal(0.0, CONFIG["noise_level"] * c)
    return float(np.clip(c, 0.0, CONFIG["peak_conc"]))  # clamp to [0, peak]


def estimate_gradient(pos) -> np.ndarray:
    """
    Finite-difference gradient using 6 symmetric probe points (+/-x, +/-y, +/-z).

        dC/dx ~= [C(x+delta) - C(x-delta)] / (2*delta)   (central difference)

    Returns 3-D gradient vector.
    """
    d   = CONFIG["sample_delta"]
    pos = np.asarray(pos, dtype=float)
    return np.array([
        (concentration(pos + [d,0,0]) - concentration(pos - [d,0,0])) / (2*d),
        (concentration(pos + [0,d,0]) - concentration(pos - [0,d,0])) / (2*d),
        (concentration(pos + [0,0,d]) - concentration(pos - [0,0,d])) / (2*d),
    ])


# ─── FULL SIMULATION (PRE-RUN ONCE) ──────────────────────────────────────────
def run_simulation():
    """
    Run complete navigation and store every frame of data.
    Called ONCE before animation so playback is smooth.

    Returns
    -------
    path      : (n+1, 3)  positions including start
    raw_conc  : (n,)      noisy sensor readings
    filt_conc : (n,)      Kalman-filtered readings
    dist_log  : (n,)      distance to source each step
    grad_log  : (n,)      gradient magnitude each step
    k_gains   : (n,)      Kalman gain K each step
    mode_log  : (n,)      'gradient' or 'random' each step
    cumeff    : (n,)      running path-efficiency % (precomputed)
    metrics   : dict      final summary statistics
    """
    # Recompute bounds here using live CONFIG values (not module-load-time defaults)
    source_in_bounds = _in_bounds(SOURCE_POS)
    start_in_bounds  = _in_bounds(START_POS)

    # Print warnings for any OOB points
    if not source_in_bounds:
        print(f"  [WARNING] Source {CONFIG['source_pos']} is OUTSIDE the "
              f"10x10x10 cube [0,{SPACE}]. Source is unreachable — result: FALSE")
    if not start_in_bounds:
        print(f"  [WARNING] Start {CONFIG['start_pos']} is OUTSIDE the "
              f"10x10x10 cube [0,{SPACE}]. Start will be clamped to nearest wall.")

    # Clamp start to boundary if out of range (nanobot begins at the nearest wall)
    pos = np.clip(START_POS.copy(), 0.0, SPACE)

    # If source is outside the cube, nanobot can never reach it — skip loop
    if not source_in_bounds:
        path = np.array([pos])
        straight = float(np.linalg.norm(SOURCE_POS - pos))
        metrics = {
            "steps": 0, "efficiency_%": 0.0,
            "final_dist": round(straight, 3),
            "conc_%": 0.0, "found": False,
        }
        # Return dummy single-step data so animation still renders
        dummy = [0.0]
        cumeff = np.array([0.0])
        return path, dummy, dummy, [straight], dummy, dummy, ["random"], cumeff, metrics

    # FIX: seed KF with first noise-free reading instead of 0.
    # Starting at 0 drags the filtered signal incorrectly for the first ~10 steps.
    init_c = concentration(pos, noise=False)
    kf = KalmanFilter1D(init_val=init_c, Q=CONFIG["kf_Q"], R=CONFIG["kf_R"])

    path      = [pos.copy()]
    raw_conc  = []
    filt_conc = []
    dist_log  = []
    grad_log  = []
    k_gains   = []
    mode_log  = []

    for _ in range(CONFIG["max_steps"]):
        dist = float(np.linalg.norm(pos - SOURCE_POS))
        if dist < CONFIG["goal_radius"]:
            # Log this final arrival position so the metrics panel
            # shows the actual arrival distance, not the step before it
            raw = concentration(pos, noise=True)
            raw_conc.append(raw)
            dist_log.append(dist)
            filtered, K = kf.step(raw)
            filt_conc.append(max(0.0, filtered))
            k_gains.append(K)
            grad_mag = float(np.linalg.norm(estimate_gradient(pos)))
            grad_log.append(grad_mag)
            mode_log.append("gradient")
            break

        # 1. Sensor reading (noisy)
        raw = concentration(pos, noise=True)
        raw_conc.append(raw)
        dist_log.append(dist)

        # 2. Kalman filter: clean the noisy reading
        filtered, K = kf.step(raw)
        filt_conc.append(max(0.0, filtered))   # FIX: clamp to >= 0
        k_gains.append(K)

        # 3. Gradient estimation via finite difference
        grad     = estimate_gradient(pos)
        grad_mag = float(np.linalg.norm(grad))
        grad_log.append(grad_mag)

        # 4. Navigation decision
        if grad_mag < 1e-6:
            # Flat zone -- random tumble (biased random walk, like E. coli)
            direction = np.random.uniform(-1.0, 1.0, 3)
            mode_log.append("random")
        else:
            direction = grad / grad_mag                   # unit gradient vector
            direction += np.random.normal(0.0, 0.03, 3)  # small noise jitter

            # FIX: guard against zero-vector after jitter
            norm = float(np.linalg.norm(direction))
            direction = (grad / grad_mag) if norm < 1e-9 else (direction / norm)
            mode_log.append("gradient")

        # 5. Move and clip to bounds
        pos = np.clip(pos + direction * CONFIG["step_size"],
                      0.0, SPACE)   # hard 10x10x10 boundary
        path.append(pos.copy())

    path = np.array(path)   # shape (n+1, 3)
    n    = len(raw_conc)

    final_dist = float(np.linalg.norm(path[-1] - SOURCE_POS))
    found      = final_dist < CONFIG["goal_radius"]

    # Edge case: zero steps (max_steps=0, source=start, or huge goal_radius)
    if n == 0 or len(path) < 2:
        raw_conc  = raw_conc  if raw_conc  else [float(concentration(path[-1], noise=False))]
        filt_conc = filt_conc if filt_conc else [raw_conc[0]]
        dist_log  = dist_log  if dist_log  else [final_dist]
        grad_log  = grad_log  if grad_log  else [0.0]
        k_gains   = k_gains   if k_gains   else [0.0]
        mode_log  = mode_log  if mode_log  else ["gradient" if found else "random"]
        cumeff    = np.array([100.0 if found else 0.0])
        metrics   = {
            "steps": n, "efficiency_%": 100.0 if found else 0.0,
            "final_dist": round(final_dist, 3),
            "conc_%": round(raw_conc[-1] / CONFIG["peak_conc"] * 100.0, 1),
            "found": found,
            "start_oob": not start_in_bounds, "source_oob": not source_in_bounds,
        }
        return (path, raw_conc, filt_conc, dist_log,
                grad_log, k_gains, mode_log, cumeff, metrics)

    # Normal: precompute per-frame cumulative efficiency
    # NOT capped at 100% — if nanobot walks less than straight line
    # (e.g. tiny step_size exhausts max_steps early), real value shown
    seg_len   = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_path  = np.cumsum(seg_len)
    straight  = float(np.linalg.norm(SOURCE_POS - np.clip(START_POS, 0, SPACE)))
    cumeff    = np.where(cum_path > 0, straight / cum_path * 100.0, 0.0) if found else np.zeros_like(cum_path)
    total_path = float(cum_path[-1]) if cum_path.size > 0 else 1.0
    efficiency = (straight / total_path * 100.0) if (total_path > 0 and found) else 0.0
    conc_pct   = round(filt_conc[-1] / CONFIG["peak_conc"] * 100.0, 1) if filt_conc else 0.0

    metrics = {
        "steps"           : n,
        "efficiency_%"    : round(efficiency, 1),
        "final_dist"      : round(final_dist, 3),
        "conc_%"          : conc_pct,
        "found"           : found,
        "start_oob"       : not start_in_bounds,
        "source_oob"      : not source_in_bounds,
    }

    return (path, raw_conc, filt_conc, dist_log,
            grad_log, k_gains, mode_log, cumeff, metrics)


# ─── ANIMATED VISUALISATION ──────────────────────────────────────────────────
def build_and_save(path, raw_conc, filt_conc, dist_log,
                   grad_log, k_gains, mode_log, cumeff, metrics):
    """
    Build a 6-panel live-updating figure and save as animated GIF.

    Panel layout (2 rows x 4 columns):
      Col 0 (both rows) : 3-D navigation path + rotating camera
      Col 1, Row 0      : Raw vs Kalman-filtered concentration
      Col 1, Row 1      : Distance to source over time
      Col 2, Row 0      : Gradient magnitude over time
      Col 2, Row 1      : Kalman gain K over time
      Col 3 (both rows) : Live numeric metrics dashboard
    """
    # Recompute bounds from live CONFIG values
    source_in_bounds = _in_bounds(SOURCE_POS)
    start_in_bounds  = _in_bounds(START_POS)

    # ── Out-of-bounds: render a single static error frame and exit ──────────────
    if not source_in_bounds:
        fig_err = plt.figure(figsize=(22, 11), facecolor=BG_C)
        fig_err.suptitle(
            "NANOBOT CHEMOTACTIC PLUME TRACKING  |  "
            "Kalman Filter + Gradient Climbing  |  "
            "Robothon 2026  |  Yash Yaduvanshi",
            color=WHITE, fontsize=11, fontweight="bold", y=0.995,
        )
        gs_err = gridspec.GridSpec(1, 2, figure=fig_err,
                                   left=0.04, right=0.97, top=0.92, bottom=0.06,
                                   wspace=0.35)

        # Left: 3D cube showing start + where source would be (clamped to wall)
        ax_3d = fig_err.add_subplot(gs_err[0, 0], projection="3d")
        for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("#222244")
        ax_3d.set_facecolor(BG_C)
        ax_3d.set_xlim(0, SPACE); ax_3d.set_ylim(0, SPACE); ax_3d.set_zlim(0, SPACE)
        ax_3d.set_xlabel("X", color=GRAY, fontsize=8)
        ax_3d.set_ylabel("Y", color=GRAY, fontsize=8)
        ax_3d.set_zlabel("Z", color=GRAY, fontsize=8)
        ax_3d.tick_params(colors=GRAY, labelsize=6)
        ax_3d.xaxis._axinfo["grid"]["color"] = "#222244"
        ax_3d.yaxis._axinfo["grid"]["color"] = "#222244"
        ax_3d.zaxis._axinfo["grid"]["color"] = "#222244"
        ax_3d.set_title("10x10x10 Bounded Space", color=WHITE, fontsize=10, pad=8)

        # Draw the nanobot start (clamped if OOB)
        clamped_start = np.clip(START_POS, 0, SPACE)
        ax_3d.scatter(*clamped_start, color=CYAN, s=120, zorder=5, label="Start (in cube)")

        # Draw a red X at the nearest wall face + annotate
        clamped_src = np.clip(SOURCE_POS, 0, SPACE)
        ax_3d.scatter(*clamped_src, color=PINK, s=300, zorder=5,
                      marker="x", linewidths=3, label="Source (OUT OF BOUNDS)")
        src_label = str(CONFIG["source_pos"])
        ax_3d.text(clamped_src[0] + 0.3, clamped_src[1] + 0.3, clamped_src[2] + 0.3,
                   f"Source {src_label} OUTSIDE CUBE",
                   color=PINK, fontsize=8, zorder=10, fontweight="bold")

        # Draw a dashed arrow from wall to where source actually is (conceptually)
        ax_3d.quiver(clamped_src[0], clamped_src[1], clamped_src[2],
                     *(SOURCE_POS - clamped_src) * 0.0,   # zero length — just marker
                     color=PINK, alpha=0.0)

        leg = ax_3d.legend(fontsize=8, loc="upper left", facecolor=PANEL, edgecolor="none")
        plt.setp(leg.get_texts(), color=WHITE)

        # Right: text panel with full explanation
        ax_msg = fig_err.add_subplot(gs_err[0, 1])
        ax_msg.set_facecolor(PANEL)
        ax_msg.axis("off")
        ax_msg.set_title("Simulation Result", color=WHITE, fontsize=12, pad=8)

        lines = [
            ("RESULT: FALSE",                      PINK,   16, "bold"),
            ("",                                    WHITE,   8, "normal"),
            ("Source point is OUTSIDE the",         WHITE,  12, "normal"),
            ("10 × 10 × 10 bounded cube.",          WHITE,  12, "normal"),
            ("",                                    WHITE,   8, "normal"),
            (f"Source given : {CONFIG['source_pos']}", GOLD, 11, "bold"),
            (f"Cube range   : [0,{SPACE}] on all axes", CYAN, 11, "normal"),
            ("",                                    WHITE,   8, "normal"),
            ("The nanobot is confined to the cube.", GRAY,  10, "normal"),
            ("It cannot exit the boundary.",         GRAY,  10, "normal"),
            ("The source is unreachable.",           GRAY,  10, "normal"),
            ("",                                    WHITE,   8, "normal"),
            ("To fix: change source_pos so all",    WHITE,  10, "normal"),
            (f"coords are within 0 to {SPACE}.",    WHITE,  10, "normal"),
        ]
        y = 0.88
        for text, color, size, weight in lines:
            ax_msg.text(0.5, y, text, ha="center", va="top",
                        color=color, fontsize=size, fontweight=weight,
                        transform=ax_msg.transAxes)
            y -= 0.065

        out = "nanobot_dynamic.gif"
        print(f"  Saving error frame as {out} ...")
        fig_err.savefig(out, facecolor=BG_C, dpi=100, format="png")
        # Save as GIF (single frame) using pillow
        from PIL import Image
        import io
        buf = io.BytesIO()
        fig_err.savefig(buf, format="png", facecolor=BG_C, dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        img.save(out, format="GIF", save_all=True, loop=0)
        print(f"  Saved: {out}  (single error frame)")
        plt.close(fig_err)
        return None

    # ── Normal simulation animation ───────────────────────────────────────────
    n_frames   = len(raw_conc)
    # Add hold frames at the end so the final state stays visible before GIF loops
    HOLD_FRAMES = 40   # ~5 seconds of pause at the end at 8 fps
    total_frames = n_frames + HOLD_FRAMES
    steps    = np.arange(n_frames)

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 11), facecolor=BG_C)
    fig.suptitle(
        "NANOBOT CHEMOTACTIC PLUME TRACKING  |  "
        "Kalman Filter + Gradient Climbing  |  "
        "Robothon 2026  |  Yash Yaduvanshi",
        color=WHITE, fontsize=11, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.52, wspace=0.40,
        left=0.04, right=0.97, top=0.95, bottom=0.07,
    )

    # Helper: apply consistent dark style to 2-D axes
    def style2d(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor("#222244")
            spine.set_linewidth(0.8)
        ax.tick_params(colors=GRAY, labelsize=7)
        ax.set_title(title, color=WHITE, fontsize=9, pad=5)
        ax.set_xlabel(xlabel, color=GRAY, fontsize=8)
        ax.set_ylabel(ylabel, color=GRAY, fontsize=8)

    # ── Panel 1: 3-D path ────────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")

    # FIX: transparent panes -- no ugly grey walls
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#222244")
    ax3d.set_facecolor(BG_C)

    ax3d.set_xlim(0, SPACE); ax3d.set_ylim(0, SPACE); ax3d.set_zlim(0, SPACE)
    ax3d.set_xlabel("X", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_ylabel("Y", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_zlabel("Z", color=GRAY, fontsize=8, labelpad=2)
    ax3d.tick_params(colors=GRAY, labelsize=6)
    ax3d.xaxis._axinfo["grid"]["color"] = "#222244"
    ax3d.yaxis._axinfo["grid"]["color"] = "#222244"
    ax3d.zaxis._axinfo["grid"]["color"] = "#222244"
    ax3d.set_title("3-D Navigation Path", color=WHITE, fontsize=10, pad=8)

    ax3d.scatter(*np.clip(START_POS, 0, SPACE), color=CYAN, s=80, zorder=5, label="Start")

    if source_in_bounds:
        ax3d.scatter(*SOURCE_POS, color=PINK, s=160, zorder=5, marker="*", label="Source")
    else:
        # Source is outside — mark the boundary wall it would be behind with a red X
        clamped_src = np.clip(SOURCE_POS, 0, SPACE)
        ax3d.scatter(*clamped_src, color=PINK, s=200, zorder=5, marker="x",
                     linewidths=2, label="Source (OUT OF BOUNDS)")
        ax3d.text(clamped_src[0], clamped_src[1], clamped_src[2],
                  f" SOURCE\nOUTSIDE\nCUBE", color=PINK, fontsize=6,
                  zorder=10, ha="left", va="bottom")

    leg = ax3d.legend(fontsize=7, loc="upper left", facecolor=PANEL, edgecolor="none")
    plt.setp(leg.get_texts(), color=WHITE)

    path_line,   = ax3d.plot([], [], [],      color=CYAN,  lw=1.4, alpha=0.85)
    nanobot_dot, = ax3d.plot([], [], [],  "o", color=GREEN, ms=8,   zorder=10)
    trail_sc      = ax3d.scatter([], [], [], c=[], cmap="plasma",
                                  s=10, alpha=0.55, vmin=0, vmax=n_frames)

    # ── Panel 2: Raw vs Filtered concentration ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style2d(ax2, "Kalman Filter: Raw vs Filtered", "Step", "Concentration")
    ax2.set_xlim(0, n_frames)
    ax2.set_ylim(0, CONFIG["peak_conc"] * 1.05)
    ax2.axhline(CONFIG["peak_conc"] * 0.9, color=GREEN, ls="--", lw=0.8, alpha=0.6)

    raw_line,  = ax2.plot([], [], color=GOLD,   lw=1.0, alpha=0.55, label="Raw (noisy)")
    filt_line, = ax2.plot([], [], color=ORANGE, lw=2.0,              label="Kalman filtered")
    raw_dot,   = ax2.plot([], [], "o", color=WHITE,  ms=5, zorder=5)
    filt_dot,  = ax2.plot([], [], "o", color=ORANGE, ms=6, zorder=5)
    leg2 = ax2.legend(fontsize=7, facecolor=BG_C, edgecolor="none")
    plt.setp(leg2.get_texts(), color=WHITE)

    # ── Panel 3: Distance to source ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style2d(ax3, "Distance to Source", "Step", "Distance (units)")
    ax3.set_xlim(0, n_frames)
    ax3.set_ylim(0, float(np.linalg.norm(SOURCE_POS - START_POS)) * 1.1)
    ax3.axhline(CONFIG["goal_radius"], color=PINK, ls="--", lw=1.2, alpha=0.9,
                label=f"Goal ({CONFIG['goal_radius']} u)")
    leg3 = ax3.legend(fontsize=7, facecolor=BG_C, edgecolor="none")
    plt.setp(leg3.get_texts(), color=WHITE)

    dist_line, = ax3.plot([], [], color=CYAN, lw=2.0)
    dist_dot,  = ax3.plot([], [], "o", color=CYAN, ms=6, zorder=5)

    # ── Panel 4: Gradient magnitude ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    style2d(ax4, "Gradient Strength  |grad C|", "Step", "|grad C|")
    ax4.set_xlim(0, n_frames)
    ax4.set_ylim(0, (max(grad_log) * 1.15) if grad_log else 10)

    grad_line, = ax4.plot([], [], color=PURP, lw=1.8)
    grad_dot,  = ax4.plot([], [], "o", color=PURP, ms=6, zorder=5)

    # ── Panel 5: Kalman gain K ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    style2d(ax5, "Kalman Gain  K = P / (P + R)", "Step", "K")
    ax5.set_xlim(0, n_frames)
    ax5.set_ylim(0, 1.05)
    ax5.axhline(0.5, color=GRAY, ls="--", lw=0.6, alpha=0.45)

    gain_line, = ax5.plot([], [], color=GREEN, lw=1.8)
    gain_dot,  = ax5.plot([], [], "o", color=GREEN, ms=6, zorder=5)

    # ── Panel 6: Live metrics dashboard ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[:, 3])
    ax6.set_facecolor(PANEL)
    ax6.axis("off")
    ax6.set_title("Live Metrics", color=WHITE, fontsize=10, pad=6)

    txt_status = ax6.text(0.5, 0.93, "", ha="center", va="top",
                          color=GREEN, fontsize=14, fontweight="bold",
                          transform=ax6.transAxes)

    metric_labels = [
        "Step", "Distance", "Raw Conc",
        "Filtered Conc", "Gradient |gradC|",
        "Kalman Gain K", "Efficiency %", "Mode",
    ]
    metric_texts = []
    y0 = 0.80
    dy = 0.095
    for idx, label in enumerate(metric_labels):
        yp = y0 - idx * dy
        ax6.text(0.06, yp, label + ":", ha="left", va="top", color=GRAY,
                 fontsize=8.5, transform=ax6.transAxes)
        t = ax6.text(0.94, yp, "-", ha="right", va="top", color=WHITE,
                     fontsize=8.5, fontweight="bold", transform=ax6.transAxes)
        metric_texts.append(t)
        ax6.plot([0.04, 0.96], [yp - 0.022, yp - 0.022],
                 color="#222244", lw=0.5, transform=ax6.transAxes)

    txt_final = ax6.text(0.5, 0.04, "", ha="center", va="bottom",
                         color=GOLD, fontsize=7.5, fontweight="bold",
                         transform=ax6.transAxes)

    # ── Animation update ──────────────────────────────────────────────────────
    def update(frame):
        # Clamp i to last real data frame during hold period
        i = min(int(frame), n_frames - 1)
        is_hold = int(frame) >= n_frames   # True during the end-pause frames

        # 3-D panel
        pts = path[:i + 2]
        path_line.set_data(pts[:, 0], pts[:, 1])
        path_line.set_3d_properties(pts[:, 2])
        nanobot_dot.set_data([pts[-1, 0]], [pts[-1, 1]])
        nanobot_dot.set_3d_properties([pts[-1, 2]])
        if i > 1:
            tp = pts[:-1]
            tc = np.linspace(0, i, len(tp))
            trail_sc._offsets3d = (tp[:, 0], tp[:, 1], tp[:, 2])
            trail_sc.set_array(tc)
        # Rotate every 2nd real frame; keep rotating slowly during hold too
        if frame % 2 == 0:
            ax3d.view_init(elev=20, azim=30 + frame * 0.20)

        # 2-D panels
        xs = steps[:i + 1]
        raw_line.set_data(xs, raw_conc[:i + 1])
        filt_line.set_data(xs, filt_conc[:i + 1])
        dist_line.set_data(xs, dist_log[:i + 1])
        grad_line.set_data(xs, grad_log[:i + 1])
        gain_line.set_data(xs, k_gains[:i + 1])
        raw_dot.set_data([i],  [raw_conc[i]])
        filt_dot.set_data([i], [filt_conc[i]])
        dist_dot.set_data([i], [dist_log[i]])
        grad_dot.set_data([i], [grad_log[i]])
        gain_dot.set_data([i], [k_gains[i]])

        # ── Metrics ───────────────────────────────────────────────────────
        mode = mode_log[i]

        # Status logic:
        # - During animation: always show SEARCHING (nanobot hasn't arrived yet)
        # - On last frame + hold: show final result from metrics["found"]
        #   (metrics["found"] uses path[-1] which is correct post-loop position)
        is_last = (i == n_frames - 1)

        if not source_in_bounds:
            txt_status.set_text("OUT OF BOUNDS!")
            txt_status.set_color(PINK)
        elif is_last or is_hold:
            if metrics["found"]:
                txt_status.set_text("SOURCE FOUND!")
                txt_status.set_color(GREEN)
            else:
                txt_status.set_text("SOURCE NOT FOUND")
                txt_status.set_color(PINK)
        else:
            txt_status.set_text("SEARCHING...")
            txt_status.set_color(CYAN)

        eff_now = cumeff[i] if i < len(cumeff) else cumeff[-1]
        vals = [
            str(i + 1),
            f"{dist_log[i]:.3f} u",
            f"{raw_conc[i]:.1f}",
            f"{filt_conc[i]:.1f}",
            f"{grad_log[i]:.4f}",
            f"{k_gains[i]:.5f}",
            f"{eff_now:.1f} %",
            mode.upper(),
        ]
        for t, v in zip(metric_texts, vals):
            t.set_text(v)
        metric_texts[7].set_color(GREEN if mode == "gradient" else ORANGE)

        # Show final summary from last real frame onward
        if is_last or is_hold:
            if not source_in_bounds:
                txt_final.set_text(
                    f"SOURCE {CONFIG['source_pos']} IS OUTSIDE\n"
                    f"THE 10x10x10 CUBE — UNREACHABLE\n"
                    f"Result: FALSE"
                )
                txt_final.set_color(PINK)
            else:
                txt_final.set_text(
                    f"Steps: {metrics['steps']}  |  "
                    f"Eff: {metrics['efficiency_%']}%  |  "
                    f"Dist: {metrics['final_dist']} u"
                )
                txt_final.set_color(GOLD)

        return (path_line, nanobot_dot,
                raw_line, filt_line, dist_line, grad_line, gain_line,
                raw_dot, filt_dot, dist_dot, grad_dot, gain_dot,
                txt_status, *metric_texts, txt_final)

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,   # real frames + hold frames
        interval=120,          # 120ms per frame = ~8fps = slow, readable
        blit=False,            # must be False -- blit=True breaks 3-D axes
        repeat=False,
    )

    out = "nanobot_dynamic.gif"
    print(f"  Saving {out} ({total_frames} frames at 8 fps) -- takes ~1 minute ...")
    ani.save(out, writer="pillow", fps=8, dpi=100)   # 8fps = slow & smooth
    print(f"  Saved: {out}")
    plt.close(fig)
    return ani


# ─── SAVE METRICS ────────────────────────────────────────────────────────────
def save_metrics(metrics: dict):
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: metrics.json")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bar = "=" * 62
    print(f"\n{bar}")
    print("  NANOBOT DYNAMIC SIMULATION -- Robothon 2026")
    print("  Kalman Filter  +  Gradient Climbing  +  Live Animation")
    print(bar)
    print(f"  Start  : {START_POS}")
    print(f"  Source : {SOURCE_POS}")
    print(f"  Kalman : Q = {CONFIG['kf_Q']}   R = {CONFIG['kf_R']}")
    print(bar + "\n")

    print("  [1/3] Running simulation ...")
    (path, raw_conc, filt_conc, dist_log,
     grad_log, k_gains, mode_log, cumeff, metrics) = run_simulation()

    print(f"        Steps      : {metrics['steps']}")
    print(f"        Efficiency : {metrics['efficiency_%']} %")
    print(f"        Found      : {metrics['found']}")
    print(f"        Final dist : {metrics['final_dist']} units\n")

    print("  [2/3] Saving metrics ...")
    save_metrics(metrics)

    print("  [3/3] Building animation ...")
    build_and_save(path, raw_conc, filt_conc, dist_log,
                   grad_log, k_gains, mode_log, cumeff, metrics)

    print(f"\n{bar}")
    print("  ALL DONE")
    print("  nanobot_dynamic.gif  -- animated GIF for judges")
    print("  metrics.json         -- performance data")
    print(f"{bar}\n")

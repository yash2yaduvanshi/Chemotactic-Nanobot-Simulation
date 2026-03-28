"""
  Author  : Yash Yaduvanshi
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import json

CONFIG = {
    "source_pos"   : [9.0, 6.0, 2.0],
    "start_pos"    : [0.4, 0.5, 0.2],
    "space_size"   : 10.0,
    "sigma"        : 3.5,
    "peak_conc"    : 1000.0,
    "step_size"    : 0.25,
    "noise_level"  : 0.03,
    "max_steps"    : 400,
    "goal_radius"  : 0.35,
    "sample_delta" : 0.3,
    "kf_Q"         : 5.0,
    "kf_R"         : 900.0,
}

SOURCE_POS = np.array(CONFIG["source_pos"])
START_POS  = np.array(CONFIG["start_pos"])
SPACE      = float(CONFIG["space_size"])

#  COLOURS 
BG    = "#0a0a1a"
PANEL = "#111133"
CYAN  = "#00e5ff"
PINK  = "#ff4081"
GREEN = "#69ff47"
GOLD  = "#ffd740"
WHITE = "#ffffff"
GRAY  = "#aaaacc"
PURP  = "#b967ff"
ORAN  = "#ff9100"

# HELPERS
def _in_bounds(pos):
    p = np.asarray(pos)
    return bool(np.all(p >= 0.0) and np.all(p <= SPACE))

class KalmanFilter1D:
    def __init__(self, init_val=0.0, Q=5.0, R=900.0):
        self.x = float(init_val)
        self.P = float(R)
        self.Q = float(Q)
        self.R = float(R)
    def step(self, z):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x, K

def concentration(pos, noise=True):
    d2 = float(np.sum((np.asarray(pos) - SOURCE_POS)**2))
    c  = CONFIG["peak_conc"] * np.exp(-d2 / (2.0 * CONFIG["sigma"]**2))
    if noise:
        c += np.random.normal(0.0, CONFIG["noise_level"] * c)
    return float(np.clip(c, 0.0, CONFIG["peak_conc"]))  # clamp to [0, peak]

def estimate_gradient(pos):
    d   = CONFIG["sample_delta"]
    pos = np.asarray(pos, dtype=float)
    return np.array([
        (concentration(pos+[d,0,0]) - concentration(pos-[d,0,0])) / (2*d),
        (concentration(pos+[0,d,0]) - concentration(pos-[0,d,0])) / (2*d),
        (concentration(pos+[0,0,d]) - concentration(pos-[0,0,d])) / (2*d),
    ])

# SIMULATION 
def run_simulation():
    src_ok   = _in_bounds(SOURCE_POS)
    start_ok = _in_bounds(START_POS)

    if not src_ok:
        print(f"[WARNING] Source {CONFIG['source_pos']} is outside cube. Result: FALSE")
    if not start_ok:
        print(f"[WARNING] Start {CONFIG['start_pos']} outside cube. Clamping to wall.")

    pos = np.clip(START_POS.copy(), 0.0, SPACE)

    if not src_ok:
        straight = float(np.linalg.norm(SOURCE_POS - pos))
        metrics  = {"steps":0,"efficiency_%":0.0,
                    "final_dist":round(straight,3),"conc_%":0.0,"found":False,
                    "start_oob": not start_ok, "source_oob": True}
        dummy  = [0.0]
        cumeff = np.array([0.0])
        return np.array([pos]), dummy, dummy, [straight], dummy, dummy, ["random"], cumeff, metrics

    init_c = concentration(pos, noise=False)
    kf     = KalmanFilter1D(init_val=init_c, Q=CONFIG["kf_Q"], R=CONFIG["kf_R"])

    path=[ pos.copy() ]; raw_c=[]; filt_c=[]; dist_l=[]; grad_l=[]; kg=[]; mode_l=[]

    for _ in range(CONFIG["max_steps"]):
        dist = float(np.linalg.norm(pos - SOURCE_POS))
        if dist < CONFIG["goal_radius"]:
            raw = concentration(pos, noise=True)
            raw_c.append(raw); dist_l.append(dist)
            fv, K = kf.step(raw)
            filt_c.append(max(0.0,fv)); kg.append(K)
            grad_l.append(float(np.linalg.norm(estimate_gradient(pos))))
            mode_l.append("gradient")
            break

        raw = concentration(pos, noise=True)
        raw_c.append(raw); dist_l.append(dist)
        fv, K = kf.step(raw)
        filt_c.append(max(0.0,fv)); kg.append(K)

        grad = estimate_gradient(pos)
        gm   = float(np.linalg.norm(grad))
        grad_l.append(gm)

        if gm < 1e-6:
            direction = np.random.uniform(-1.0,1.0,3); mode_l.append("random")
        else:
            direction = grad/gm
            direction += np.random.normal(0.0,0.03,3)
            nrm = float(np.linalg.norm(direction))
            direction = (grad/gm) if nrm < 1e-9 else (direction/nrm)
            mode_l.append("gradient")

        pos = np.clip(pos + direction*CONFIG["step_size"], 0.0, SPACE)
        path.append(pos.copy())

    path = np.array(path)
    n    = len(raw_c)

    # Edge case: zero steps taken (max_steps=0, or source=start, or goal_radius huge)
    # In these cases path has only 1 point → np.diff gives empty → handle gracefully
    if n == 0 or len(path) < 2:
        final_dist = float(np.linalg.norm(path[-1] - SOURCE_POS))
        found      = final_dist < CONFIG["goal_radius"]
        # Return single dummy data point so animation renders cleanly
        raw_c   = raw_c   if raw_c   else [float(concentration(path[-1], noise=False))]
        filt_c  = filt_c  if filt_c  else [raw_c[0]]
        dist_l  = dist_l  if dist_l  else [final_dist]
        grad_l  = grad_l  if grad_l  else [0.0]
        kg      = kg      if kg      else [0.0]
        mode_l  = mode_l  if mode_l  else ["gradient" if found else "random"]
        cumeff  = np.array([100.0 if found else 0.0])
        metrics = {
            "steps": n, "efficiency_%": 100.0 if found else 0.0,
            "final_dist": round(final_dist, 3),
            "conc_%": round(raw_c[-1]/CONFIG["peak_conc"]*100.0, 1),
            "found": found,
            "start_oob": not start_ok, "source_oob": not src_ok,
        }
        return path, raw_c, filt_c, dist_l, grad_l, kg, mode_l, cumeff, metrics

    # Compute found FIRST — needed for efficiency calculation below
    final_dist = float(np.linalg.norm(path[-1] - SOURCE_POS))
    found      = final_dist < CONFIG["goal_radius"]

    seg    = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum    = np.cumsum(seg)
    st     = float(np.linalg.norm(SOURCE_POS - np.clip(START_POS, 0, SPACE)))
    # Efficiency = 0 if not found, real value if found
    cumeff = np.where(cum > 0, st / cum * 100.0, 0.0) if found else np.zeros_like(cum)
    total  = float(cum[-1]) if cum.size > 0 else 1.0
    eff    = (st / total * 100.0) if (total > 0 and found) else 0.0

    conc_pct = round(filt_c[-1] / CONFIG["peak_conc"] * 100.0, 1) if filt_c else 0.0

    metrics = {
        "steps": n, "efficiency_%": round(eff, 1),
        "final_dist": round(final_dist, 3),
        "conc_%": conc_pct,
        "found": found,
        "start_oob": not start_ok, "source_oob": not src_ok,
    }
    return path, raw_c, filt_c, dist_l, grad_l, kg, mode_l, cumeff, metrics


# INTERACTIVE LIVE VIEWER 
def show_live(path, raw_conc, filt_conc, dist_log,
              grad_log, k_gains, mode_log, cumeff, metrics):

    n      = len(raw_conc)
    steps  = np.arange(n)
    HOLD   = 60   # hold frames at end

    # ── Figure: 3 rows x 4 cols ───────────────────────────────────────────────
    # Row 0-1 col 0 : 3D path (tall)
    # Row 0   col 1 : raw vs filtered
    # Row 1   col 1 : distance
    # Row 0   col 2 : gradient
    # Row 1   col 2 : kalman gain
    # Row 0-1 col 3 : metrics
    # Row 2   col 0-3 : view-angle buttons bar

    fig = plt.figure(figsize=(22, 12), facecolor=BG)
    fig.suptitle(
        "NANOBOT CHEMOTACTIC PLUME TRACKING  |  "
        "Kalman Filter + Gradient Climbing  |  "
        "Robothon 2026  |  Yash Yaduvanshi  |  "
        "← DRAG 3D PANEL TO ROTATE",
        color=WHITE, fontsize=10, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        height_ratios=[1, 1, 0.12],
        hspace=0.52, wspace=0.38,
        left=0.04, right=0.97, top=0.955, bottom=0.04,
    )

    def style2d(ax, title, xl, yl):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#222244"); sp.set_linewidth(0.8)
        ax.tick_params(colors=GRAY, labelsize=7)
        ax.set_title(title, color=WHITE, fontsize=9, pad=4)
        ax.set_xlabel(xl, color=GRAY, fontsize=8)
        ax.set_ylabel(yl, color=GRAY, fontsize=8)

    # ── Panel 1: 3D (rows 0-1, col 0) ────────────────────────────────────────
    ax3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#222244")
    ax3d.set_facecolor(BG)
    ax3d.set_xlim(0,SPACE); ax3d.set_ylim(0,SPACE); ax3d.set_zlim(0,SPACE)
    ax3d.set_xlabel("X", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_ylabel("Y", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_zlabel("Z", color=GRAY, fontsize=8, labelpad=2)
    ax3d.tick_params(colors=GRAY, labelsize=6)
    ax3d.xaxis._axinfo["grid"]["color"] = "#222244"
    ax3d.yaxis._axinfo["grid"]["color"] = "#222244"
    ax3d.zaxis._axinfo["grid"]["color"] = "#222244"
    ax3d.set_title("3-D Path  ←drag to rotate→", color=WHITE, fontsize=9, pad=6)

    ax3d.scatter(*np.clip(START_POS,0,SPACE), color=CYAN, s=80, zorder=5, label="Start")
    if _in_bounds(SOURCE_POS):
        ax3d.scatter(*SOURCE_POS, color=PINK, s=160, zorder=5, marker="*", label="Source")
    else:
        csr = np.clip(SOURCE_POS,0,SPACE)
        ax3d.scatter(*csr, color=PINK, s=200, marker="x", linewidths=2, zorder=5,
                     label="Source (OOB)")
        ax3d.text(csr[0],csr[1],csr[2], " OOB", color=PINK, fontsize=7)

    leg = ax3d.legend(fontsize=7, loc="upper left", facecolor=PANEL, edgecolor="none")
    plt.setp(leg.get_texts(), color=WHITE)

    path_line,   = ax3d.plot([],[],[], color=CYAN, lw=1.4, alpha=0.85)
    nanobot_dot, = ax3d.plot([],[],[], "o", color=GREEN, ms=8, zorder=10)
    trail_sc      = ax3d.scatter([],[],[], c=[], cmap="plasma",
                                  s=10, alpha=0.55, vmin=0, vmax=n)

    # ── Panel 2: Raw vs Filtered ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style2d(ax2, "Kalman: Raw vs Filtered", "Step", "Concentration")
    ax2.set_xlim(0,n); ax2.set_ylim(0, CONFIG["peak_conc"]*1.05)
    ax2.axhline(CONFIG["peak_conc"]*0.9, color=GREEN, ls="--", lw=0.8, alpha=0.6)
    raw_line,  = ax2.plot([],[], color=GOLD,  lw=1.0, alpha=0.55, label="Raw")
    filt_line, = ax2.plot([],[], color=ORAN,  lw=2.0,              label="Kalman filtered")
    raw_dot,   = ax2.plot([],[], "o", color=WHITE, ms=5, zorder=5)
    filt_dot,  = ax2.plot([],[], "o", color=ORAN,  ms=6, zorder=5)
    leg2 = ax2.legend(fontsize=7, facecolor=BG, edgecolor="none")
    plt.setp(leg2.get_texts(), color=WHITE)

    # Panel 3: Distance
    ax3 = fig.add_subplot(gs[1, 1])
    style2d(ax3, "Distance to Source", "Step", "Distance (u)")
    ax3.set_xlim(0,n)
    ax3.set_ylim(0, float(np.linalg.norm(SOURCE_POS - np.clip(START_POS,0,SPACE)))*1.1)
    ax3.axhline(CONFIG["goal_radius"], color=PINK, ls="--", lw=1.2,
                label=f"Goal ({CONFIG['goal_radius']} u)")
    leg3 = ax3.legend(fontsize=7, facecolor=BG, edgecolor="none")
    plt.setp(leg3.get_texts(), color=WHITE)
    dist_line, = ax3.plot([],[], color=CYAN, lw=2.0)
    dist_dot,  = ax3.plot([],[], "o", color=CYAN, ms=6, zorder=5)

    # Panel 4: Gradient
    ax4 = fig.add_subplot(gs[0, 2])
    style2d(ax4, "Gradient Strength  |grad C|", "Step", "|grad C|")
    ax4.set_xlim(0,n)
    ax4.set_ylim(0, (max(grad_log)*1.15) if grad_log else 10)
    grad_line, = ax4.plot([],[], color=PURP, lw=1.8)
    grad_dot,  = ax4.plot([],[], "o", color=PURP, ms=6, zorder=5)

    # Panel 5: Kalman Gain
    ax5 = fig.add_subplot(gs[1, 2])
    style2d(ax5, "Kalman Gain  K = P/(P+R)", "Step", "K")
    ax5.set_xlim(0,n); ax5.set_ylim(0, 1.05)
    ax5.axhline(0.5, color=GRAY, ls="--", lw=0.6, alpha=0.45)
    gain_line, = ax5.plot([],[], color=GREEN, lw=1.8)
    gain_dot,  = ax5.plot([],[], "o", color=GREEN, ms=6, zorder=5)

    # Panel 6: Live metrics (rows 0-1, col 3)
    ax6 = fig.add_subplot(gs[0:2, 3])
    ax6.set_facecolor(PANEL); ax6.axis("off")
    ax6.set_title("Live Metrics", color=WHITE, fontsize=10, pad=6)

    txt_status = ax6.text(0.5, 0.93, "", ha="center", va="top",
                          color=GREEN, fontsize=13, fontweight="bold",
                          transform=ax6.transAxes)

    mlabels = ["Step","Distance","Raw Conc","Filtered Conc",
               "Gradient |gradC|","Kalman Gain K","Efficiency %","Mode"]
    mtexts = []
    y0, dy = 0.80, 0.095
    for idx, lb in enumerate(mlabels):
        yp = y0 - idx*dy
        ax6.text(0.06, yp, lb+":", ha="left", va="top", color=GRAY,
                 fontsize=8, transform=ax6.transAxes)
        t = ax6.text(0.94, yp, "-", ha="right", va="top", color=WHITE,
                     fontsize=8, fontweight="bold", transform=ax6.transAxes)
        mtexts.append(t)
        ax6.plot([0.04,0.96],[yp-0.02,yp-0.02],
                 color="#222244", lw=0.5, transform=ax6.transAxes)

    txt_final = ax6.text(0.5, 0.04, "", ha="center", va="bottom",
                         color=GOLD, fontsize=7.5, fontweight="bold",
                         transform=ax6.transAxes)

    # View-angle buttons — hardcoded figure coords, no canvas.draw() needed
    # Buttons sit at the very bottom of the figure under the 3D panel.
    # Figure is 22in wide. Left edge of 3D panel is at left=0.04 (from gs).
    # Each button: width=0.045, height=0.04, gap=0.005
    # Label text sits above the 4 buttons using fig.text() in figure coords.

    view_presets = [
        ("Front (XZ)", 0,   0),
        ("Side (YZ)",  0,  90),
        ("Top (XY)",  90,   0),
        ("Iso (3D)",  25,  45),
    ]

    # "Snap view:" label — placed in figure coords just above the buttons
    fig.text(0.04, 0.068, "Snap view:", color=GOLD,
             fontsize=8, fontweight="bold", va="bottom")

    _buttons = []
    btn_left  = 0.04    # start x (same as gs left)
    btn_width = 0.047
    btn_height= 0.042
    btn_gap   = 0.006
    btn_bottom= 0.018   # y position — clear of figure edge

    for k, (label, elev, azim) in enumerate(view_presets):
        bx = btn_left + k * (btn_width + btn_gap)
        btn_ax = fig.add_axes([bx, btn_bottom, btn_width, btn_height])
        btn = Button(btn_ax, label, color="#1a1a44", hovercolor="#333388")
        btn.label.set_color(WHITE)
        btn.label.set_fontsize(8)
        btn.label.set_fontweight("bold")

        def make_cb(e, a):
            def cb(_):
                ax3d.view_init(elev=e, azim=a)
                fig.canvas.draw_idle()
            return cb

        btn.on_clicked(make_cb(elev, azim))
        _buttons.append(btn)

    # Animation update
    def update(frame):
        i       = min(int(frame), n-1)
        is_hold = int(frame) >= n

        # 3D
        pts = path[:i+2]
        path_line.set_data(pts[:,0], pts[:,1])
        path_line.set_3d_properties(pts[:,2])
        nanobot_dot.set_data([pts[-1,0]], [pts[-1,1]])
        nanobot_dot.set_3d_properties([pts[-1,2]])
        if i > 1:
            tp = pts[:-1]
            trail_sc._offsets3d = (tp[:,0], tp[:,1], tp[:,2])
            trail_sc.set_array(np.linspace(0, i, len(tp)))
        # Only auto-rotate when user is not dragging
        # (auto-rotation is disabled here so user can drag freely)

        # 2D panels
        xs = steps[:i+1]
        raw_line.set_data(xs, raw_conc[:i+1])
        filt_line.set_data(xs, filt_conc[:i+1])
        dist_line.set_data(xs, dist_log[:i+1])
        grad_line.set_data(xs, grad_log[:i+1])
        gain_line.set_data(xs, k_gains[:i+1])
        raw_dot.set_data([i],  [raw_conc[i]])
        filt_dot.set_data([i], [filt_conc[i]])
        dist_dot.set_data([i], [dist_log[i]])
        grad_dot.set_data([i], [grad_log[i]])
        gain_dot.set_data([i], [k_gains[i]])

        # Metrics
        mode = mode_log[i]
        is_last = (i == n-1)

        if not _in_bounds(SOURCE_POS):
            txt_status.set_text("OUT OF BOUNDS!"); txt_status.set_color(PINK)
        elif is_last or is_hold:
            if metrics["found"]:
                txt_status.set_text("SOURCE FOUND!"); txt_status.set_color(GREEN)
            else:
                txt_status.set_text("SOURCE NOT FOUND"); txt_status.set_color(PINK)
        else:
            txt_status.set_text("SEARCHING..."); txt_status.set_color(CYAN)

        eff_now = cumeff[i] if i < len(cumeff) else cumeff[-1]
        vals = [
            str(i+1),
            f"{dist_log[i]:.3f} u",
            f"{raw_conc[i]:.1f}",
            f"{filt_conc[i]:.1f}",
            f"{grad_log[i]:.4f}",
            f"{k_gains[i]:.5f}",
            f"{eff_now:.1f} %",
            mode.upper(),
        ]
        for t, v in zip(mtexts, vals):
            t.set_text(v)
        mtexts[7].set_color(GREEN if mode == "gradient" else ORAN)

        if is_last or is_hold:
            txt_final.set_text(
                f"Steps: {metrics['steps']}  |  "
                f"Eff: {metrics['efficiency_%']}%  |  "
                f"Dist: {metrics['final_dist']} u"
            )
            txt_final.set_color(GREEN if metrics["found"] else PINK)

        return (path_line, nanobot_dot, raw_line, filt_line,
                dist_line, grad_line, gain_line,
                raw_dot, filt_dot, dist_dot, grad_dot, gain_dot,
                txt_status, *mtexts, txt_final)

    ani = animation.FuncAnimation(
        fig, update,
        frames=n + HOLD,
        interval=150,      # 150ms per frame — smooth and readable
        blit=False,
        repeat=False,
    )

    print("  Live viewer running.")
    print("  Drag the 3D panel to rotate from any angle.")
    print("  Use the Front/Side/Top/Iso buttons to snap to preset views.")
    print("  Close the window to exit.\n")

    plt.show()
    return ani


# MAIN
if __name__ == "__main__":
    bar = "=" * 60
    print(f"\n{bar}")
    print("  NANOBOT LIVE INTERACTIVE VIEWER — Robothon 2026")
    print(f"{bar}")
    print(f"  Start  : {START_POS}")
    print(f"  Source : {SOURCE_POS}")
    print(f"{bar}\n")

    print("  Running simulation ...")
    result = run_simulation()
    path, raw_conc, filt_conc, dist_log, grad_log, k_gains, mode_log, cumeff, metrics = result

    print(f"  Steps      : {metrics['steps']}")
    print(f"  Efficiency : {metrics['efficiency_%']} %")
    print(f"  Found      : {metrics['found']}")
    print(f"  Final dist : {metrics['final_dist']} units\n")

    show_live(path, raw_conc, filt_conc, dist_log,
              grad_log, k_gains, mode_log, cumeff, metrics)

#!/usr/bin/env python3
# Granulobot Script 2 — paper-based bias voltage model (Eqs. 1–3) with robust startup
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Arrow
from matplotlib.animation import FuncAnimation, FFMpegWriter
from copy import deepcopy

# =========================
# USER CONTROLS
# =========================
n = 7
gear_radius = 1              # [m] drawn circle radius; center spacing d0 = 2*gear_radius

# Inputs per gear: bias voltage u [V], motor constant k [N·m/V], translational mass m [kg]
gear_params = [
    {"bias_u": 1, "k": 0.229, "m": 2.0},  # 1
    {"bias_u": 1, "k": 0.229, "m": 2.0},  # 2
    {"bias_u": 1, "k": 0.229, "m": 2.0},  # 3
    {"bias_u": 1, "k": 0.229, "m": 2.0},  # 4
    {"bias_u": 1, "k": 0.229, "m": 2.0},  # 5
    {"bias_u": 1, "k": 0.229, "m": 2.0},  # 6
    {"bias_u": 1, "k": 0.229, "m": 2.0},  # 7
]
assert len(gear_params) == n

# Motor/drive constants
eta0     = 0.231      # [N·m·s/rad] viscous motor constant
Gamma_f  = 0.024      # [N·m] Coulomb friction (static/kinetic threshold)
alpha    = -1       # dimensionless motor feedback (1+alpha ~ 0.1 => low effective damping)
G        = .5       # [N·m/rad] weak torsional spring gain
I  = 7.5e-1     # [kg·m^2] moment of inertia
rot_rat  = 0.5          # Magnet-to-gear shell moment of inertia ratio

scale = 15
damp_coeff =  .5

# Neighbor coupling (kept light so the drive can win at startup)
k_tan = 1 * scale   # [N·m/(rad/s)] viscous penalty (ω_i+ω_j)
k_norm      = 1 * scale
k_m     = 1 * scale        # Magnet force constant for actuation load (harmonic, unphysical)
k_eq    = 1 * scale
zeta_n     = k_norm * damp_coeff       # Set higher than magnet force
zeta_t     = k_tan * damp_coeff
zeta_m     = k_m * damp_coeff
mu = .5 # Static friction Coulomb threshold 

# Constraints & integration
dt              = 1e-3
steps_per_frame = 33
frames          = 900
fps             = 30
neighbor_iters  = 6    # SHAKE iterations for neighbor links
global_iters    = 2    # SHAKE iterations for non-overlap
w_clip          = 12.0 # [rad/s] safety clamp on |ω|

# Data and metrics variables
data = {"Time": []}

# =========================
d0   = 2.0 * gear_radius
u    = np.array([p["bias_u"] for p in gear_params], dtype=float)
karr = np.array([p["k"]      for p in gear_params], dtype=float)
mtr  = np.array([p["m"]      for p in gear_params], dtype=float)

phi = np.zeros(n)
theta = np.zeros(n)
slip = np.zeros(n)
omega_theta = np.zeros(n)
omega_phi = np.zeros(n)
t     = 0.0

# positions around a circle
R = d0 / (2 * np.sin(np.pi / n))
ang0 = np.linspace(0, 2*np.pi, n, endpoint=False)
r = np.column_stack([R*np.cos(ang0), R*np.sin(ang0)])
v = np.zeros_like(r)
r_i = r.copy()
delta = np.zeros((n, n, 2))
for j in range(n):
    delta[:, j] = r_i - r
    r_i = np.roll(r_i, -1, 0)
Ftan = np.zeros((n, n))
Fnorm = np.zeros((n, n))

F_gear_net = np.zeros((n, 2))
Fmag = np.zeros((n, 2))


# =========================
# HELPERS
# =========================
def neighbor_pairs():
    for i in range(n):
        yield i, (i+1) % n
phi_presets = np.array([np.arctan2(delta[i][1][1], delta[i][1][0]) for i in range(n)])

def compute_gear_deltas(v, omega_theta, delta):
    """Return Tgear (torques from friction spring), Ftan (tangential forces), and Fnorm (normal forces2)"""
    Tgear = np.zeros(n)
    d_tan    = np.zeros((n, n))
    d_norm = np.zeros((n, n))

    d = delta.copy()
    dist = np.linalg.norm(d, axis=2)
    e = d / dist[:, :, np.newaxis]
    t_hat = np.roll(e, 1, 2)
    t_hat[:, :, 0] = -t_hat[:, :, 0]
    relative_v = np.zeros_like(delta)
    omega_slip = np.zeros_like(d_tan)

    for i in range(1, n):
        relative_v[:, i] = np.roll(v, -i, 0) - v
        omega_shift = np.roll(omega_theta, -i)
        omega_slip[:, i] = omega_theta + omega_shift

       
        d_norm[:, i] = np.vecdot(relative_v[:, i], e[:, i])
        d_tan[:, i] = -omega_slip[:, i] * (dist[:, i]) / 2 + np.vecdot(relative_v[:, i], t_hat[:, i])
    return d_tan, d_norm, e, t_hat, relative_v, omega_slip

def compute_gear_torque_and_forces(d_tan, d_norm, Ftan, Fnorm, e, t_hat):
    Ftan += d_tan * k_tan * dt
    Fnorm += d_norm * k_norm * dt
    Ftan = coulomb_clip(Ftan, Fnorm)
    tot_tan = Ftan + zeta_t * d_tan
    tot_norm = Fnorm + zeta_n * d_norm
    Fgear = tot_tan[:, :, np.newaxis] * t_hat + tot_norm[:, :, np.newaxis] * e
    Tgear = tot_tan * (Fnorm + d0)
    # print(tot_tan[:, 1])
    return Ftan, Fnorm, Fgear, Tgear


def coulomb_clip(Ftan, Fnorm):
    Ftan[:, 2:n-1] = np.where(np.abs(Ftan[:, 2:n-1]) > mu * np.abs(Fnorm[:, 2:n-1]), 
                              sgn_nonzero(Ftan[:, 2:n-1], 1) * mu * np.abs(Fnorm[:, 2:n-1]),
                              Ftan[:, 2:n-1])
    return Ftan

def no_overlaps(delta, Fgear, Tgear, Ftan, Fnorm):
    mask = np.linalg.norm(delta, axis=2) >= d0
    mask[:, 0] = np.array(n * [True])
    mask[:, 1] = np.array(n * [False])
    mask[:, n-1] = np.array(n * [False])
    Fgear[mask] = np.zeros_like(Fgear[0, 0])
    Tgear[mask], Ftan[mask], Fnorm[mask] = (0, 0, 0)
    return Fgear, Tgear, Ftan, Fnorm, mask

def compute_magnet_torque_and_forces(delta_1, phi, omega_phi):
    """Return Fmag (force from magnets) and Tmag (load torque, fed into Gamma_l)"""
    Fmag = np.zeros_like(r)
    Tmag = np.zeros(n)
    mag = np.column_stack([2*gear_radius*np.cos(phi), 2*gear_radius*np.sin(phi)])
    tan_mag = np.column_stack([-gear_radius*np.sin(phi), gear_radius*np.cos(phi)])

    mag_disp = delta_1 - mag

    forward_f = mag_disp * k_m - omega_phi[:, np.newaxis] * tan_mag * zeta_m
    Fmag = forward_f - np.roll(forward_f, 1, 0)

    # calculate torque via cross product of magnet with only forward force
    Tmag = np.vecdot(forward_f, tan_mag)

    return Fmag, Tmag


    

def sgn_nonzero(val, fallback):
    sign = val >= 0
    neg = np.where(sign, val, -1)
    sign = neg > 0
    return np.where(sign, 1, neg)

# --- Logging setup ---
log_columns = ["time", "theta", "omega_theta", "phi", "omega_phi", "r", "v", "F_gear_net", "Fmag", "Ftan", "Fnorm", "relative_v", "relative_omega"]
log_df = pd.DataFrame(columns=log_columns)

def flatten(var):
    """Convert scalars/arrays/lists into something storable in CSV."""
    if np.isscalar(var):
        return var
    elif isinstance(var, (list, np.ndarray)):
        return np.array(var).flatten().tolist()
    else:
        return str(var)

def log_state(t, theta, omega_theta, phi, omega_phi, r, v, F_gear_net, Fmag, Ftan, Fnorm, relative_v, relative_omega):
    global log_df

    row = {
        "time": t,
        "theta": flatten(theta),
        "omega_theta": flatten(omega_theta),
        "phi": flatten(phi),
        "omega_phi": flatten(omega_phi),
        "r": flatten(r),
        "v": flatten(v),
        "F_gear_net": flatten(F_gear_net),
        "Fmag": flatten(Fmag),
        "Ftan": Ftan.copy(),
        "Fnorm": Fnorm.copy(),
        "relative_v": relative_v.copy(),
        "relative_omega": relative_omega.copy()
    }
    log_df.loc[len(log_df)] = row

# Data I need to add: velocity differences
# Ideas: add Coulomb friction


# =========================
#(Eqs. 1–3 with inertia)
# =========================
def step():
    global theta, omega_theta, phi, omega_phi, r, v, t, delta, Ftan, Fnorm, F_gear_net, Fmag

    t0 = deepcopy(t)
    ang_sgn = sgn_nonzero(theta-phi, 1)
    vel_sgn = sgn_nonzero(omega_theta-omega_phi, 1)
    # velocity-Verlet: calculate positions first and take all velocities a half-step back
    t       += dt
    r       += v * dt
    theta   += omega_theta * dt
    phi     += omega_phi * dt

    r_i = r.copy()
    for j in range(n):
        delta[:, j] = r_i - r
        r_i = np.roll(r_i, -1, 0)
    
    # neighbor torques/forces from slip
    d_tan, d_norm, e, t_hat, relative_v, omega_slip = compute_gear_deltas(v, omega_theta, delta)
    Ftan, Fnorm, Fgear, Tgear = compute_gear_torque_and_forces(d_tan, d_norm, Ftan, Fnorm, e, t_hat)
    Fgear, Tgear, Ftan, Fnorm, mask1 = no_overlaps(delta, Fgear, Tgear, Ftan, Fnorm)
    F_gear_net = np.sum(Fgear, 1)
    T_gear_net = np.sum(Tgear, 1)

    Fmag, Tmag = compute_magnet_torque_and_forces(delta[:, 1].copy(), phi+phi_presets, omega_phi)

    # Eq. (2): U = u - (alpha * eta0 / k) * ω - (G / k) * θ  (units-consistent)
    U = ang_sgn * u - (alpha * eta0 / karr) * (omega_theta - omega_phi) - (G / karr) * (theta-phi)

    # drive torque
    T_drive = karr * U

    # Total torque should be Tgear + Tmag. Assume Tgear equally distributed by moment of inertia. We widen the gap
    # by the actuation resistance torque.
    Gamma_l = T_gear_net - Tmag
    mask = np.abs(Gamma_l + T_drive) <= Gamma_f
    T_res = T_drive - Gamma_f * vel_sgn - eta0 * (omega_theta - omega_phi)
    torque_adjust = k_eq * np.where(mask, eta0 * (omega_phi - omega_theta), Gamma_l + T_res) - Gamma_l

    T_theta = T_gear_net + torque_adjust / 2
    T_phi = Tmag - torque_adjust / 2

    alpha_phi = T_phi / (rot_rat * mtr * I)
    alpha_omega = T_theta / ((1-rot_rat) * mtr * I)
    omega_theta += alpha_omega * dt
    omega_phi += alpha_phi * dt

    # translate (tangential forces + damping)
    F = F_gear_net + Fmag
    a = F / mtr[:, None]
    v += a * dt

    # log state
    log_state(t, theta, omega_theta, phi, omega_phi, r, v, F_gear_net, Fmag, Ftan, Fnorm, relative_v, omega_slip)

    if(int(25 * t) != int(25 * t0)):
        print("t", t)
    
    # print("t", t, "||Fgear, Fmag||", T_gear_net)



# ANIMATION
# =========================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal', 'box')
L = (R + 1.5*d0)
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
ax.set_title("Granulobot ring — bias-voltage model (Eqs 1–3)")

patches, spokes, couplings, force_gear, force_mag = [], [], [], [], []

def directions():
    directions = np.array([0.0, 0.0] * n).reshape(n, 2)
    for i in range(n):
        dir = phi[i]+phi_presets[i]
        directions[i] = np.array([np.cos(dir), np.sin(dir)])
    return directions
    

for i,(x0,y0) in enumerate(r):
    direct = directions()
    c = Circle((x0,y0), gear_radius, fill=False, lw=2)
    l = Line2D([x0,x0+gear_radius * direct[i][0]],[y0,y0+gear_radius * direct[i][1]], lw=2) # phi direction: magnet
    d = Line2D([x0,x0+gear_radius * direct[i][0]],[y0,y0+gear_radius * direct[i][1]], lw=1, color=(1.0, 0.0, 0.0)) # theta direction: gear
    fg = Arrow(x0, y0, 0, 0, width=0.1, color=(1.0, 0.5, 0.5), linestyle='dotted')
    fm = Arrow(x0, y0, 0, 0, width=0.1, color=(0.5, 0.5, 1.0), linestyle='dotted')
    ax.add_patch(c); ax.add_line(l); ax.add_line(d), ax.add_patch(fg), ax.add_patch(fm)
    ax.text
    patches.append(c); spokes.append(l); couplings.append(d); force_gear.append(fg), force_mag.append(fm)

time_text = ax.text(
    0.05, 0.95, '',
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

def clip_vector(v, max_length):
    norm = np.linalg.norm(v)
    if norm > max_length:
        return v / norm * max_length
    return v

def animate(_):
    for _ in range(steps_per_frame): step()
    direct = directions()
    for i,(c,s,d,fg,fm) in enumerate(zip(patches,spokes,couplings, force_gear, force_mag)):
        x,y=r[i]
        c.center=(x,y)
        xx=x+gear_radius*(np.cos(phi_presets[i] + phi[i]))
        yy=y+gear_radius*(np.sin(phi_presets[i] + phi[i]))
        s.set_data([x,xx],[y,yy])
        x1=x+gear_radius*(np.cos(phi_presets[i] + theta[i]))
        y1=y+gear_radius*(np.sin(phi_presets[i] + theta[i]))
        d.set_data([x,x1],[y,y1])
        fg0 = clip_vector(F_gear_net[i], 50)
        fm0 = clip_vector(Fmag[i], 50)
        fg.set_data(x, y, fg0[0], fg0[1])
        fm.set_data(x, y, fm0[0], fm0[1])
        # ax.annotate("", xytext=(x,y), xy=(x+F_gear_net[i][0], 
        #                                   y+F_gear_net[i][1]))
        
    time_text.set_text(f"t = {t:.2f}")
    return patches+spokes+couplings+[time_text]

anim = FuncAnimation(fig, animate, frames=frames, blit=True)
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save("../logs/granulobots_C-S.mp4", writer=writer)
log_df.to_csv("../logs/simulation_log.csv", index=False)
print("Saved granulobots_script2.mp4")

#!/usr/bin/env python3
"""textrun.py - turn text into a GPS-art running activity (GPX).

The route spells the text with a blocky one-stroke font. Strokes are traced as
several parallel passes (out along one edge, back along the other) so letters
look thick. The "run" has correlated, soft-capped pace variation, hill and
corner slowdowns, stops, a position-based elevation field, GPS jitter, and
lagged heart rate / cadence / temperature.

Intended for private test uploads (set the activity to "Only You").

    python3 textrun.py "Testing some features" -o run.gpx --preview run.png
"""
import argparse
import math
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

# ---------------------------------------------------------------- font ----
# Grid: x 0..2, y 0..4 (y up). Each glyph is a list of polylines.
GLYPHS = {
    "A": [[(0, 0), (0, 3), (1, 4), (2, 3), (2, 0)], [(0, 2), (2, 2)]],
    "B": [[(0, 0), (0, 4), (1.5, 4), (2, 3.5), (2, 2.5), (1.5, 2), (0, 2)],
          [(1.5, 2), (2, 1.5), (2, 0.5), (1.5, 0), (0, 0)]],
    "C": [[(2, 4), (0, 4), (0, 0), (2, 0)]],
    "D": [[(0, 0), (0, 4), (1, 4), (2, 3), (2, 1), (1, 0), (0, 0)]],
    "E": [[(2, 4), (0, 4), (0, 0), (2, 0)], [(0, 2), (1.5, 2)]],
    "F": [[(2, 4), (0, 4), (0, 0)], [(0, 2), (1.5, 2)]],
    "G": [[(2, 4), (0, 4), (0, 0), (2, 0), (2, 2), (1, 2)]],
    "H": [[(0, 0), (0, 4)], [(2, 0), (2, 4)], [(0, 2), (2, 2)]],
    "I": [[(0, 4), (2, 4)], [(1, 4), (1, 0)], [(0, 0), (2, 0)]],
    "J": [[(2, 4), (2, 0), (0, 0), (0, 1)]],
    "K": [[(0, 0), (0, 4)], [(2, 4), (0, 2), (2, 0)]],
    "L": [[(0, 4), (0, 0), (2, 0)]],
    "M": [[(0, 0), (0, 4), (1, 2), (2, 4), (2, 0)]],
    "N": [[(0, 0), (0, 4), (2, 0), (2, 4)]],
    "O": [[(0, 0), (0, 4), (2, 4), (2, 0), (0, 0)]],
    "P": [[(0, 0), (0, 4), (2, 4), (2, 2), (0, 2)]],
    "Q": [[(0, 0), (0, 4), (2, 4), (2, 0), (0, 0)], [(1, 1), (2, 0)]],
    "R": [[(0, 0), (0, 4), (2, 4), (2, 2), (0, 2)], [(1, 2), (2, 0)]],
    "S": [[(2, 4), (0, 4), (0, 2), (2, 2), (2, 0), (0, 0)]],
    "T": [[(0, 4), (2, 4)], [(1, 4), (1, 0)]],
    "U": [[(0, 4), (0, 0), (2, 0), (2, 4)]],
    "V": [[(0, 4), (1, 0), (2, 4)]],
    "W": [[(0, 4), (0.5, 0), (1, 2), (1.5, 0), (2, 4)]],
    "X": [[(0, 4), (2, 0)], [(0, 0), (2, 4)]],
    "Y": [[(0, 4), (1, 2), (2, 4)], [(1, 2), (1, 0)]],
    "Z": [[(0, 4), (2, 4), (0, 0), (2, 0)]],
    "0": [[(0, 0), (0, 4), (2, 4), (2, 0), (0, 0)], [(0, 0), (2, 4)]],
    "1": [[(0, 3), (1, 4), (1, 0)]],
    "2": [[(0, 4), (2, 4), (2, 2), (0, 2), (0, 0), (2, 0)]],
    "3": [[(0, 4), (2, 4), (2, 0), (0, 0)], [(0, 2), (2, 2)]],
    "4": [[(0, 4), (0, 2), (2, 2)], [(2, 4), (2, 0)]],
    "5": [[(2, 4), (0, 4), (0, 2), (2, 2), (2, 0), (0, 0)]],
    "6": [[(2, 4), (0, 4), (0, 0), (2, 0), (2, 2), (0, 2)]],
    "7": [[(0, 4), (2, 4), (1, 0)]],
    "8": [[(0, 0), (0, 4), (2, 4), (2, 0), (0, 0)], [(0, 2), (2, 2)]],
    "9": [[(2, 0), (2, 4), (0, 4), (0, 2), (2, 2)]],
    "-": [[(0, 2), (2, 2)]],
    ".": [[(1, 0), (1, 0.5)]],
    "!": [[(1, 4), (1, 1.2)], [(1, 0), (1, 0.4)]],
    ":": [[(1, 3), (1, 2.6)], [(1, 1.4), (1, 1)]],
}

MI = 1609.344


def wrap(text, max_chars):
    lines = []
    for chunk in text.upper().split("|"):
        cur = ""
        for w in chunk.split():
            if cur and len(cur) + 1 + len(w) > max_chars:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        if cur:
            lines.append(cur)
    return lines


def order_strokes(strokes, cur):
    rem = [np.array(s, float) for s in strokes]
    out = []
    if cur is None:
        first = rem.pop(0)
        out.append(first)
        cur = first[-1]
    while rem:
        best = None
        for i, s in enumerate(rem):
            for rev in (False, True):
                p = s[-1] if rev else s[0]
                d = np.hypot(*(p - cur))
                if best is None or d < best[0]:
                    best = (d, i, rev)
        _, i, rev = best
        s = rem.pop(i)
        s = s[::-1] if rev else s
        out.append(s)
        cur = s[-1]
    return out


def offset_polyline(P, d):
    """Offset a polyline by d to its left, with clamped miter joins."""
    n = len(P)
    if n < 2 or d == 0:
        return P.copy()
    seg = P[1:] - P[:-1]
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
    nrm = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    out = np.empty_like(P)
    out[0] = P[0] + nrm[0] * d
    out[-1] = P[-1] + nrm[-1] * d
    for i in range(1, n - 1):
        m = nrm[i - 1] + nrm[i]
        ml = np.hypot(*m)
        if ml < 1e-6:                      # 180 degree reversal
            out[i] = P[i] + nrm[i - 1] * d
            continue
        m /= ml
        cos_half = max(float(np.dot(m, nrm[i])), 0.5)
        out[i] = P[i] + m * d / cos_half
    return out


def thicken(stroke, w, passes):
    """Trace a stroke as `passes` parallel lines, alternating direction."""
    if passes <= 1 or w <= 0:
        return [stroke]
    out = []
    for k, off in enumerate(np.linspace(-w, w, passes)):
        line = offset_polyline(stroke, off)
        out.append(line if k % 2 == 0 else line[::-1])
    return out


def unit_path(lines, w, passes):
    adv = 3.0 + 2 * w
    space = 2.0 + 2 * w
    pitch = 6.5 + 2 * w
    widths = [sum(space if c == " " else adv for c in l) - (1.0 + 2 * w) for l in lines]
    W = max(widths)
    pts = []
    for li, l in enumerate(lines):
        ox = (W - widths[li]) / 2 + w
        oy = (len(lines) - 1 - li) * pitch
        for c in l:
            if c == " ":
                ox += space
                continue
            if c not in GLYPHS:
                print(f"warning: no glyph for {c!r}, skipped", file=sys.stderr)
                continue
            cur = pts[-1] if pts else None
            for st in order_strokes(GLYPHS[c], cur):
                for line in thicken(st, w, passes):
                    for p in line:
                        q = np.array([ox + p[0], oy + p[1]])
                        if not pts or np.hypot(*(q - pts[-1])) > 1e-9:
                            pts.append(q)
            ox += adv
    return np.array(pts)


def fillet(P, r, n=6):
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        l1, l2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
        d = min(r, 0.45 * l1, 0.45 * l2)
        A = b - (b - a) / l1 * d
        B = b + (c - b) / l2 * d
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * A + 2 * t * (1 - t) * b + t ** 2 * B)
    out.append(P[-1])
    return np.array(out)


def resample(P, ds=1.0):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    N = max(int(round(L / ds)), 2)
    s = np.linspace(0, L, N + 1)
    return np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1]), L / N


def smooth(a, k):
    k = max(int(k) | 1, 1)
    pad = np.pad(a, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


# ------------------------------------------------------------- terrain ----
class Terrain:
    def __init__(self, rng, hilliness, base):
        n = 5
        lam = np.exp(rng.uniform(np.log(120), np.log(1000), n))
        ang = rng.uniform(0, 2 * np.pi, n)
        k = 2 * np.pi / lam
        self.kx, self.ky = np.cos(ang) * k, np.sin(ang) * k
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.amp = rng.uniform(0.012, 0.03, n) * hilliness / k
        self.base = base

    def z(self, x, y):
        x, y = np.asarray(x)[:, None], np.asarray(y)[:, None]
        return self.base + (self.amp * np.sin(self.kx * x + self.ky * y + self.ph)).sum(1)


def ar1(rng, tau, sigma):
    a = math.exp(-1.0 / tau)
    sd = sigma * math.sqrt(1 - a * a)
    state = rng.normal() * sigma

    def step():
        nonlocal state
        state = a * state + sd * rng.normal()
        return state
    return step


# ------------------------------------------------------------ simulate ----
def build(args, path_len, v0, seed):
    rng = np.random.default_rng(seed)
    w = args.thickness_width
    passes = args.thickness
    U = unit_path(wrap(args.text, args.max_chars), w, passes)
    Lu = np.linalg.norm(np.diff(U, axis=0), axis=1).sum()
    P = U * (path_len / Lu)
    P = fillet(P, args.corner_radius)
    X, Y, ds = resample(P)
    N = len(X)
    L = ds * (N - 1)

    # terrain + grade (capped)
    terr = Terrain(rng, args.hilliness, args.base_elev)
    Z = terr.z(X, Y)
    g = np.gradient(smooth(Z, 15), ds)
    g99 = np.percentile(np.abs(g), 99.5)
    if g99 > args.max_grade:
        terr.amp *= args.max_grade / g99
        Z = terr.z(X, Y)
        g = np.gradient(smooth(Z, 15), ds)

    # corner speed limit (with braking lookahead)
    th = np.unwrap(np.arctan2(np.diff(Y), np.diff(X)))
    kappa = smooth(np.abs(np.diff(th)) / ds, 5)
    kappa = np.concatenate([[kappa[0]], kappa, [kappa[-1]]])
    lim = np.minimum(np.sqrt(args.lat_accel / np.maximum(kappa, 1e-3)), 6.0)
    lim = np.maximum(lim, 2.2)
    lim[-1] = min(lim[-1], 1.2)
    a_dec, a_acc = 1.4, 1.0
    for i in range(N - 2, -1, -1):
        lim[i] = min(lim[i], math.sqrt(lim[i + 1] ** 2 + 2 * a_dec * ds))

    # pace noise: slow + medium + fast OU, soft-capped in log space
    wts = np.array([0.5, 0.75, 0.45])
    sig = args.pace_var * wts / np.sqrt((wts ** 2).sum())
    ou = [ar1(rng, tau, s) for tau, s in zip((300, 45, 6), sig)]
    cap = args.pace_cap

    stops = sorted(rng.uniform(0.08 * L, 0.95 * L, args.stops))
    gps_x, gps_y = ar1(rng, 30, 0.9), ar1(rng, 30, 0.9)
    baro = ar1(rng, 10, 0.3)
    drift_sd = 0.02
    hr_noise, cad_noise, tmp_noise = ar1(rng, 8, 1.2), ar1(rng, 6, 1.5), ar1(rng, 120, 0.4)
    drift, temp0 = 0.0, args.temp + rng.uniform(-0.5, 0.5)

    hr = args.hr_start - rng.uniform(35, 50)
    ev = 0.0
    s = v = 0.0
    hold = 0
    t = 0
    rows = []
    while True:
        i = min(int(s / ds), N - 1)
        x = sum(f() for f in ou)
        fac = math.exp(cap * math.tanh(x / cap))
        hill = min(max(1 - 2.2 * g[i], 0.8), 1.18)
        warm = 0.88 + 0.12 * min(1.0, t / 180)
        fat = 1 - args.fatigue * s / L
        vdes = min(v0 * fac * hill * warm * fat, lim[i])
        if hold > 0:
            vdes, hold = 0.0, hold - 1
        elif stops and s >= stops[0]:
            stops.pop(0)
            hold = int(rng.integers(4, 14))
            vdes = 0.0
        v += min(max(vdes - v, -a_dec), a_acc)
        s = min(L, s + v)

        # sensors
        ev += (v - ev) / 12
        load = max((ev / v0) ** 1.4 + 4.0 * g[i], 0.6)
        base_hr = args.hr_start + (args.hr_end - args.hr_start) * (s / L) ** 0.8
        ss = base_hr + 38 * (load - 1)
        tau = 22 if ss > hr else 40
        hr += (ss - hr) / tau
        cad = 0.0 if v < 0.3 else args.cadence * (v / v0) ** 0.4 + cad_noise()
        drift += rng.normal() * drift_sd
        drift *= 0.999

        j = min(s / ds, N - 1.0000001)
        j0 = int(j)
        fr = j - j0
        px = X[j0] * (1 - fr) + X[j0 + 1] * fr
        py = Y[j0] * (1 - fr) + Y[j0 + 1] * fr
        rows.append((t, px + gps_x() + rng.normal() * 0.12,
                     py + gps_y() + rng.normal() * 0.12,
                     px, py, v, hr + hr_noise(), cad,
                     temp0 + tmp_noise() + 0.6 * math.sin(t / 900)))
        t += 1
        if s >= L - 1e-9:
            break
        if t > 6 * 3600:
            break

    R = np.array(rows)
    zt = terr.z(R[:, 3], R[:, 4])
    elev = zt + np.array([baro() for _ in range(len(R))]) + np.cumsum(rng.normal(0, 0.02, len(R))) * 0.5
    elev = np.round(elev, 1)
    lat0, lon0 = args.lat, args.lon
    lat = lat0 + R[:, 2] / 111320.0
    lon = lon0 + R[:, 1] / (111320.0 * math.cos(math.radians(lat0)))
    hrv = np.clip(np.round(R[:, 6]), 60, args.hr_max).astype(int)
    cadv = np.clip(np.round(R[:, 7]), 0, 230).astype(int)
    return dict(t=R[:, 0], lat=lat, lon=lon, ele=elev, hr=hrv, cad=cadv,
                temp=np.round(R[:, 8]).astype(int), x=R[:, 1], y=R[:, 2], ds=ds, ref=len(U))


def haversine(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def generate(args, seed):
    target_m = args.miles * MI
    target_v = MI / args.pace_s
    path_len, v0 = target_m, target_v
    for _ in range(10):
        r = build(args, path_len, v0, seed)
        d = haversine(r["lat"], r["lon"]).sum()
        va = d / r["t"][-1]
        ed, ev_ = target_m / d, target_v / va
        if abs(1 - ed) < 0.002 and abs(1 - ev_) < 0.004:
            break
        path_len *= ed
        v0 *= ev_
    return r


# -------------------------------------------------------------- output ----
def fmt_pace(v):
    if v <= 0:
        return "--"
    sec = MI / v
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def write_gpx(r, start, path, name, full_cad):
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<gpx version="1.1" creator="textrun" '
                'xmlns="http://www.topografix.com/GPX/1/1" '
                'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">\n'
                f'<metadata><time>{start:%Y-%m-%dT%H:%M:%SZ}</time></metadata>\n'
                f'<trk><name>{name}</name><type>running</type><trkseg>\n')
        for i in range(len(r["t"])):
            ts = start + timedelta(seconds=float(r["t"][i]))
            cad = int(r["cad"][i]) if full_cad else int(round(r["cad"][i] / 2))
            f.write(f'<trkpt lat="{r["lat"][i]:.7f}" lon="{r["lon"][i]:.7f}">'
                    f'<ele>{r["ele"][i]:.1f}</ele><time>{ts:%Y-%m-%dT%H:%M:%SZ}</time>'
                    '<extensions><gpxtpx:TrackPointExtension>'
                    f'<gpxtpx:atemp>{r["temp"][i]}</gpxtpx:atemp>'
                    f'<gpxtpx:hr>{r["hr"][i]}</gpxtpx:hr><gpxtpx:cad>{cad}</gpxtpx:cad>'
                    '</gpxtpx:TrackPointExtension></extensions></trkpt>\n')
        f.write('</trkseg></trk></gpx>\n')


def report(r):
    step = haversine(r["lat"], r["lon"])
    d = step.sum()
    T = r["t"][-1]
    sp = smooth(step, 30)
    ele = smooth(r["ele"], 15)
    dz = np.diff(ele)
    grade = np.abs(np.gradient(ele)[:-1] / np.maximum(step, 0.5))
    print(f"distance      {d / MI:.3f} mi   time {int(T // 60)}:{int(T % 60):02d}   avg pace {fmt_pace(d / T)}/mi")
    print(f"pace (30s)    fastest {fmt_pace(sp.max())}  slowest(moving) {fmt_pace(sp[sp > 0.8].min())}  "
          f"max step speed {step.max():.2f} m/s")
    print(f"elevation     {r['ele'].min():.1f}..{r['ele'].max():.1f} m, gain {dz[dz > 0].sum():.0f} m, "
          f"p99 grade {np.percentile(smooth(grade, 15), 99) * 100:.1f}%")
    print(f"heart rate    {r['hr'].min()}..{r['hr'].max()} bpm   cadence avg {r['cad'][r['cad'] > 0].mean():.0f} spm")
    print(f"stopped secs  {(step < 0.3).sum()}   points {len(r['t'])}")


def preview(r, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    step = haversine(r["lat"], r["lon"])
    t = r["t"][1:] / 60
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(r["x"], r["y"], lw=0.8, color="#d9480f")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, smooth(MI / np.maximum(step, 0.3) / 60, 15), lw=0.8)
    a1.set_ylim(12, 6)
    a1.set_ylabel("min/mi")
    a2 = fig.add_subplot(gs[1, 1])
    a2.plot(r["t"] / 60, r["ele"], lw=0.8)
    a2.set_ylabel("elev (m)")
    a3 = fig.add_subplot(gs[2, 1])
    a3.plot(r["t"] / 60, r["hr"], lw=0.8, color="crimson")
    a3.set_ylabel("HR")
    a3.set_xlabel("min")
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("-o", "--out", default="run.gpx")
    ap.add_argument("--preview")
    ap.add_argument("--miles", type=float, default=5.0)
    ap.add_argument("--pace", default="9:00", help="average pace per mile, m:ss")
    ap.add_argument("--lat", type=float, default=40.0150)
    ap.add_argument("--lon", type=float, default=-105.2705)
    ap.add_argument("--start", help="ISO UTC start time; default ends now")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-chars", type=int, default=10, help="wrap width; use | for manual line breaks")
    ap.add_argument("--thickness", type=int, default=3, help="parallel passes per stroke (1 = thin line)")
    ap.add_argument("--thickness-width", type=float, default=0.3, help="half-width of the stroke, in glyph units")
    ap.add_argument("--corner-radius", type=float, default=5.0, help="meters")
    ap.add_argument("--lat-accel", type=float, default=2.5, help="m/s^2; lower slows corners more")
    ap.add_argument("--pace-var", type=float, default=0.07, help="relative pace std-dev")
    ap.add_argument("--pace-cap", type=float, default=0.16, help="soft cap on pace deviation")
    ap.add_argument("--fatigue", type=float, default=0.03)
    ap.add_argument("--stops", type=int, default=3)
    ap.add_argument("--hilliness", type=float, default=1.0)
    ap.add_argument("--max-grade", type=float, default=0.07)
    ap.add_argument("--base-elev", type=float, default=1655.0)
    ap.add_argument("--hr-start", type=float, default=140)
    ap.add_argument("--hr-end", type=float, default=165)
    ap.add_argument("--hr-max", type=int, default=190)
    ap.add_argument("--cadence", type=float, default=172, help="steps/min at target pace")
    ap.add_argument("--full-cadence", action="store_true", help="write steps/min instead of per-foot rpm")
    ap.add_argument("--temp", type=float, default=15)
    args = ap.parse_args()

    m, s = args.pace.split(":")
    args.pace_s = int(m) * 60 + int(s)
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 10**6)
    args.text = args.text
    r = generate(args, seed)
    if args.start:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        start = (datetime.now(timezone.utc) - timedelta(seconds=float(r["t"][-1]) + 120)).replace(microsecond=0)
    write_gpx(r, start, args.out, args.text.replace("|", " ").title(), args.full_cadence)
    print(f"seed {seed} -> {args.out}")
    report(r)
    if args.preview:
        preview(r, args.preview, args.text)
        print(f"preview -> {args.preview}")


if __name__ == "__main__":
    main()

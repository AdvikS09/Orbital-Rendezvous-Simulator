"""Representative time histories and the common-command comparison.

Inputs: results/histories/*.npz (make_histories.py) and
        results/mechanisms/common_command/ (mechanisms.py common)
Usage:  python experiments/ext/fig_histories.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_utils as RU  # noqa: E402

H = os.path.join(RU.RESULTS, "histories")
CC = os.path.join(RU.RESULTS, "mechanisms", "common_command")


def sat_intervals(t, raw, amax):
    s = np.linalg.norm(raw, axis=1) > amax
    out, k = [], 0
    while k < s.size:
        if s[k]:
            j = k
            while j + 1 < s.size and s[j + 1]:
                j += 1
            out.append((t[k], t[j]))
            k = j + 1
        else:
            k += 1
    return out


def representative():
    import matplotlib.pyplot as plt
    idx = pd.read_csv(os.path.join(H, "index.csv"))
    picks = []
    rep = idx[idx.meta_label.fillna("").str.contains("representative saturated")]
    if len(rep):
        picks.append((rep.sid.iloc[0], "saturated, both arrive"))
    for cls, lab in (("nl_only", "CW fails, reference arrives"), ("cw_only", "CW arrives, reference fails")):
        q = idx[idx.outcome_terminal_cw == cls]
        if len(q):
            picks.append((q.sid.iloc[0], lab))
    fig, axs = plt.subplots(4, len(picks), figsize=(7.4, 7.0), sharex="col", squeeze=False)
    data = []
    for j, (sid, lab) in enumerate(picks):
        z = np.load(os.path.join(H, f"{sid}.npz"))
        scn = json.loads(str(z["scenario_json"]))
        am = scn["a_max"]
        for m, col, ls, name in (("cw", RU.BLUE, "-", "CW"), ("eci", RU.VIOLET, "--", "nonlinear reference")):
            t, S = z[f"{m}_t"], z[f"{m}_S"]
            axs[0, j].semilogy(t, np.maximum(np.linalg.norm(S[:, :3], axis=1), 1e-6), color=col, ls=ls, lw=1.2, label=name)
            axs[1, j].semilogy(t, np.maximum(np.linalg.norm(S[:, 3:], axis=1), 1e-9), color=col, ls=ls, lw=1.2)
            axs[2, j].plot(t, np.linalg.norm(z[f"{m}_app"], axis=1), color=col, ls=ls, lw=1.2)
            for k in range(0, t.size, 10):
                data.append({"sid": sid, "model": m, "t_s": t[k], "range_m": float(np.linalg.norm(S[k, :3])),
                             "speed_mps": float(np.linalg.norm(S[k, 3:])), "acc_mps2": float(np.linalg.norm(z[f"{m}_app"][k]))})
        for a, b in sat_intervals(z["cw_t"], z["cw_raw"], am):
            for r in range(4):
                axs[r, j].axvspan(a, b, color=RU.GRAY_LIGHT, alpha=0.35, lw=0)
            data.append({"sid": sid, "model": "cw_saturated_interval", "t_s": a, "t_end_s": b})
        n = min(z["cw_t"].size, z["eci_t"].size)
        d = np.linalg.norm(z["cw_S"][:n, :3] - z["eci_S"][:n, :3], axis=1)
        for k in range(0, n, 10):
            data.append({"sid": sid, "model": "cw_minus_reference", "t_s": z["cw_t"][k], "pos_diff_m": float(d[k])})
        axs[3, j].semilogy(z["cw_t"][:n], np.maximum(d, 1e-9), color=RU.INK, lw=1.0)
        T = scn["horizon_orbits"] * 2 * np.pi / np.sqrt(3.986004418e14 / (6378137.0 + scn["altitude"]) ** 3)
        for r in (0, 1):
            axs[r, j].axvline(T - 300, color=RU.MUTED, lw=0.7, ls=":")
        axs[0, j].axhline(1.0, color=RU.MUTED, lw=0.7)
        axs[1, j].axhline(0.01, color=RU.MUTED, lw=0.7)
        axs[2, j].axhline(am, color=RU.MUTED, lw=0.7)
        ax_ = scn["meta"].get("axis", "")
        axs[0, j].set_title(f"{RU.AXIS_LABEL.get(ax_, ax_)} {scn['meta'].get('sep_m', 0)/1e3:.3f} km, "
                            f"$\\omega_n$={scn['wn_ratio']:g}n, $a_{{max}}$={am:g}\n{lab}", fontsize=7)
        axs[3, j].set_xlabel("time [s]")
    axs[0, 0].set_ylabel("range [m]")
    axs[1, 0].set_ylabel("relative speed [m/s]")
    axs[2, 0].set_ylabel("|applied acc.| [m/s$^2$]")
    axs[3, 0].set_ylabel("|CW - reference| [m]")
    axs[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.text(0.01, 0.003, "shaded: CW loop saturated; dotted: T - 300 s (latest admissible arrival); "
             "horizontal lines: 1 m, 0.01 m/s, a_max", fontsize=6.5, color=RU.INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    RU.save(fig, "fig_representative_histories", pd.DataFrame(data))


def common_command():
    import matplotlib.pyplot as plt
    summ = pd.read_csv(os.path.join(CC, "common_command_summary.csv"))
    picks = summ[summ.meta_label.isin(["representative saturated", "disc_gt_1m upper bracket end"]) |
                 summ.meta_label.fillna("").str.contains("nl_terminal lower")].head(3)
    fig, axs = plt.subplots(2, len(picks), figsize=(7.4, 4.4), sharex="col", squeeze=False)
    data = []
    for j, r in enumerate(picks.itertuples()):
        z = np.load(os.path.join(CC, r.history_file))
        t = z["cl_cw_t"]
        n1 = min(t.size, z["cc_eci_t"].size)
        n2 = min(t.size, z["cl_eci_t"].size)
        d_cc = np.linalg.norm(z["cc_eci_S"][:n1, :3] - z["cl_cw_S"][:n1, :3], axis=1)
        d_cl = np.linalg.norm(z["cl_eci_S"][:n2, :3] - z["cl_cw_S"][:n2, :3], axis=1)
        du = np.linalg.norm(z["cl_eci_app"][:n2] - z["cl_cw_app"][:n2], axis=1)
        axs[0, j].semilogy(t[:n1], np.maximum(d_cc, 1e-9), color=RU.ORANGE, ls="--", label="same command replayed (open loop)")
        axs[0, j].semilogy(t[:n2], np.maximum(d_cl, 1e-9), color=RU.BLUE, label="independent closed loops")
        axs[1, j].semilogy(t[:n2], np.maximum(du, 1e-12), color=RU.BLUE)
        axs[0, j].set_title(f"{RU.AXIS_LABEL.get(r.meta_axis, r.meta_axis)} {r.sep_m/1e3:.2f} km, "
                            f"$\\omega_n$={r.wn_ratio:g}n, a$_{{max}}$={r.a_max:g}", fontsize=7)
        axs[1, j].set_xlabel("time [s]")
        for k in range(0, n2, 10):
            data.append({"sid": r.sid, "t_s": t[k], "closed_loop_diff_m": d_cl[k],
                         "common_command_diff_m": d_cc[k] if k < n1 else np.nan, "cmd_diff_mps2": du[k]})
    axs[0, 0].set_ylabel("|nonlinear - CW| position [m]")
    axs[1, 0].set_ylabel("|command difference| [m/s$^2$]")
    axs[0, 0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    RU.save(fig, "fig_common_command", pd.DataFrame(data))
    cols = [c for c in summ.columns if c.startswith(("cc_eci", "cl_eci", "masking_ratio_eci", "cc_nerm_vs", "cc_so_vs", "cl_so_vs"))]
    t = summ[["meta_axis", "sep_m", "wn_ratio", "a_max", "meta_label"] + cols].copy()
    RU.write_table(t, "tab_common_command", "Common-command propagation (diagnostic): the CW closed loop's applied "
                   "acceleration history replayed open-loop through each nonlinear model (cc_*), compared with the "
                   "independent closed loops (cl_*); masking ratio = peak replay difference / peak closed-loop "
                   "difference. Not an additive decomposition.", "tab:common",
                   fmt={c: "{:.4g}" for c in cols})


def main():
    RU.setup()
    if os.path.exists(os.path.join(H, "index.csv")):
        representative()
    if os.path.exists(os.path.join(CC, "common_command_summary.csv")):
        common_command()


if __name__ == "__main__":
    main()

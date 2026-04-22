import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import json
from data_logger import parse_file
from vizualizacija import parsed_packets_to_class, sestavi_podatke, ID_ACC, ID_GYRO, ID_MAG
from scipy.signal import resample

BIN_FILE  = "LOG06.bin"
JSON_FILE = "LOG06_oznake.json"

ACTIVITY = ["tipkanje", "mirovanje", "telefon", "miska", "premik"]
COLOURS = {
    "tipkanje": "green",
    "mirovanje": "blue",
    "telefon": "red",
    "miska": "orange",
    "premik": "purple",
}

def label(bin_file, json_out):

    print(f"Nalaganje: {bin_file}")

    packets = parse_file(bin_file)
    class_packets = parsed_packets_to_class(packets)

    acc_packets  = [p for p in class_packets if p.id == ID_ACC]
    gyro_packets = [p for p in class_packets if p.id == ID_GYRO]
    mag_packets  = [p for p in class_packets if p.id == ID_MAG]


    Fvz_acc,  sig_acc  = sestavi_podatke(acc_packets)
    Fvz_gyro, sig_gyro = sestavi_podatke(gyro_packets)
    Fvz_mag,  sig_mag  = sestavi_podatke(mag_packets)

    t_acc  = np.arange(len(sig_acc))  / Fvz_acc
    t_gyro = np.arange(len(sig_gyro)) / Fvz_gyro
    t_mag  = np.arange(len(sig_mag))  / Fvz_mag

    print(f"ACC:  {len(sig_acc)}  vzorcev @ {Fvz_acc:.1f}  Hz, trajanje = {t_acc[-1]:.1f}s")
    print(f"GYRO: {len(sig_gyro)} vzorcev @ {Fvz_gyro:.1f} Hz, trajanje = {t_gyro[-1]:.1f}s")
    print(f"MAG:  {len(sig_mag)}  vzorcev @ {Fvz_mag:.1f}  Hz, trajanje = {t_mag[-1]:.1f}s")

    labels           = []
    current_activity = [ACTIVITY[0]]

    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=False)
    plt.subplots_adjust(hspace=0.35)

    ax[0].plot(t_acc,  sig_acc[:,  0], linewidth=0.7, label="X", color="tab:red")
    ax[0].plot(t_acc,  sig_acc[:,  1], linewidth=0.7, label="Y", color="tab:green")
    ax[0].plot(t_acc,  sig_acc[:,  2], linewidth=0.7, label="Z", color="tab:blue")
    ax[0].set_ylabel("Pospeskometer [g]", fontsize=9)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t_gyro, sig_gyro[:, 0], linewidth=0.7, label="X", color="tab:red")
    ax[1].plot(t_gyro, sig_gyro[:, 1], linewidth=0.7, label="Y", color="tab:green")
    ax[1].plot(t_gyro, sig_gyro[:, 2], linewidth=0.7, label="Z", color="tab:blue")
    ax[1].set_ylabel("Giroskop [°/s]", fontsize=9)
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t_mag,  sig_mag[:,  0], linewidth=0.7, label="X", color="tab:red")
    ax[2].plot(t_mag,  sig_mag[:,  1], linewidth=0.7, label="Y", color="tab:green")
    ax[2].plot(t_mag,  sig_mag[:,  2], linewidth=0.7, label="Z", color="tab:blue")
    ax[2].set_ylabel("Magnetometer [mGauss]", fontsize=9)
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].grid(True, alpha=0.3)

    ax[2].set_xlabel("Cas [s]")

    def update_title():
        activity = current_activity[0]
        colour   = COLOURS[activity]
        keys_str = "  ".join([f"[{i+1}]{a}" for i, a in enumerate(ACTIVITY)])

        if labels:
            zadnje = "  |  ".join([f"{l['activity']} {l['t_start']:.1f}-{l['t_end']:.1f}s" for l in labels[-3:]])
        else:
            zadnje = "/"

        ax[0].set_title(
            f"{bin_file}  |  Activity: [{activity.upper()}]  |  Labels: {len(labels)}\n"
            f"{keys_str}  |  [Z] undo\n"
            f"Zadnje: {zadnje}",
            color=colour, fontsize=9
        )

        for spine in ax[0].spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2)

        fig.canvas.draw_idle()

    update_title()

    def on_key(event):
        keys = {str(i + 1): a for i, a in enumerate(ACTIVITY)}

        if event.key in keys:
            current_activity[0] = keys[event.key]
            update_title()

        if event.key == "z" and labels:
            last = labels.pop()
            for a in ax:
                if a.patches:
                    a.patches[-1].remove()
            if ax[0].texts:
                ax[0].texts[-1].remove()

            print(f"Undo: {last['activity']} | {last['t_start']:.2f}s | {last['t_end']:.2f}s")
            update_title()
            fig.canvas.draw_idle()

    def snap_to_nearest(t, threshold=0.5):
        for l in labels:
            if abs(t - l["t_start"]) < threshold:
                return l["t_start"]
            if abs(t - l["t_end"]) < threshold:
                return l["t_end"]
        return t

    def make_on_select(source_ax, source_fvz):

        def on_select(t_min, t_max):
            if t_max - t_min < 0.05:
                return

            t_min = snap_to_nearest(t_min)
            t_max = snap_to_nearest(t_max)

            t_min = max(0.0, t_min)
            t_max = min(t_acc[-1], t_max)  

            if t_max - t_min < 0.05:
                return
            
            for l in labels:
                if not (t_max <= l["t_start"] or t_min >= l["t_end"]):
                    print(f"Prekrivanje z [{l['activity']}] {l['t_start']:.2f}s – {l['t_end']:.2f}s")
                    return

            activity = current_activity[0]
            colour = COLOURS[activity]

            labels.append({
                "activity":     activity,
                "t_start":      round(t_min, 3),
                "t_end":        round(t_max, 3),
                "i_start_acc":  int(t_min * Fvz_acc),
                "i_end_acc":    int(t_max * Fvz_acc),
                "i_start_gyro": int(t_min * Fvz_gyro),
                "i_end_gyro":   int(t_max * Fvz_gyro),
                "i_start_mag":  int(t_min * Fvz_mag),
                "i_end_mag":    int(t_max * Fvz_mag),
            })

            ax[0].axvspan(t_min, t_max, alpha=0.25, color=colour)
            ax[1].axvspan(t_min, t_max, alpha=0.25, color=colour)
            ax[2].axvspan(t_min, t_max, alpha=0.25, color=colour)
            
            mid = (t_min + t_max) / 2
            ax[0].text(
                mid, ax[0].get_ylim()[1] * 0.88,
                activity[:3].upper(),
                ha="center", fontsize=7,
                color=colour, fontweight="bold"
            )

            update_title()
            print(f" [{activity}] {t_min:.2f}s | {t_max:.2f}s")

        return on_select

    span_acc  = SpanSelector(ax[0], make_on_select(ax[0], Fvz_acc),  "horizontal",
                             useblit=True, props=dict(alpha=0.15, facecolor="yellow"))
    span_gyro = SpanSelector(ax[1], make_on_select(ax[1], Fvz_gyro), "horizontal",
                             useblit=True, props=dict(alpha=0.15, facecolor="yellow"))
    span_mag  = SpanSelector(ax[2], make_on_select(ax[2], Fvz_mag),  "horizontal",
                             useblit=True, props=dict(alpha=0.15, facecolor="yellow"))

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("NAVODILA:")
    for i, a in enumerate(ACTIVITY):
        print(f"  [{i+1}] → {a}")
    print("[Z] - undo zadnje oznake")
    print("Oznaci interval na grafu")

    plt.tight_layout()
    plt.show()
    
    result = {
        "file":      bin_file,
        "Fvz_acc":   round(Fvz_acc, 2),
        "Fvz_gyro":  round(Fvz_gyro, 2),
        "Fvz_mag":   round(Fvz_mag, 2),
        "duration":  round(float(t_acc[-1]), 2),
        "n_samples_acc":  len(sig_acc),
        "n_samples_gyro": len(sig_gyro),
        "n_samples_mag":  len(sig_mag),
        "activity":  ACTIVITY,
        "labels":    sorted(labels, key=lambda x: x["t_start"]),
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nShranjeno: {json_out}")
    print(f"Skupaj oznak: {len(labels)}")
    for a in ACTIVITY:
        n = sum(1 for l in labels if l["activity"] == a)
        if n > 0:
            print(f"{a}: {n}x")

if __name__ == "__main__":
    label(BIN_FILE, JSON_FILE)
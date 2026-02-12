# graphs.py — Square plots + dual legends, show AND save to PDF
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# --- PDF con testo selezionabile ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,   # titolo meno invasivo
    "axes.labelsize": 18,
    "legend.fontsize": 14,
})

# ------------------------ Dati ------------------------
overhead = {"NB_diag": 1.1, "LDA": 1.6, "FisherMix": 2.2, "P-Hyper": 2.6, "QDA_full": 4.0}

# ImageNet-1K
acc_imagenet_c10   = {"NB_diag": 78.84, "LDA": 86.05, "FisherMix": 84.74, "P-Hyper": 85.74, "QDA_full": 84.40}
acc_imagenet_c100c = {"NB_diag": 25.40, "LDA": 37.60, "FisherMix": 40.10, "P-Hyper": 39.80, "QDA_full": 64.30}

# Places365 (dal paper)
acc_places_c10     = {"NB_diag": 78.85, "LDA": 86.26, "FisherMix": 86.56, "P-Hyper": 86.31, "QDA_full": 85.12}
acc_places_c100c   = {"NB_diag": 15.11, "LDA": 25.99, "FisherMix": 29.08, "P-Hyper": 38.64, "QDA_full": 46.54}

methods = ["NB_diag", "LDA", "FisherMix", "P-Hyper", "QDA_full"]
colors  = {"NB_diag": "#1f77b4", "LDA": "#ff7f0e", "FisherMix": "#2ca02c", "P-Hyper": "#9467bd", "QDA_full": "#d62728"}
markers = {"NB_diag": "o", "LDA": "s", "FisherMix": "^", "P-Hyper": "D", "QDA_full": "P"}
line_styles = {"ImageNet-1K": "-", "Places365": "--"}
line_colors = {"ImageNet-1K": "#2c2c2c", "Places365": "#7a7a7a"}

# ------------------------ Stile assi ------------------------
def stylize_axes(ax):
    ax.set_facecolor("#f7f7f9")
    for s in ax.spines.values():
        s.set_color("#c8c8d0"); s.set_linewidth(1.0)
    ax.minorticks_on()
    ax.grid(which="major", color="#cfd3dc", linestyle="-", linewidth=0.85, alpha=0.75)
    ax.grid(which="minor", color="#e5e8ef", linestyle=":", linewidth=0.75, alpha=0.85)
    ax.tick_params(axis="both", which="major", length=5, width=1.0, color="#7a7a7a")
    ax.tick_params(axis="both", which="minor", length=3, width=0.7, color="#9a9a9a")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# ------------------------ Plot + salvataggio ------------------------
def plot_acc_vs_overhead(dataset_name, acc_img, acc_plc, save_pdf=False, outdir="./figures", slug=None):
    fig, ax = plt.subplots(figsize=(8,6), dpi=140)  # quadrato
    methods_sorted = sorted(methods, key=lambda m: overhead[m])
    x = [overhead[m] for m in methods_sorted]
    y_img = [acc_img[m] for m in methods_sorted]
    y_plc = [acc_plc[m] for m in methods_sorted]

    lw, ms = 2.6, 10

    # punti (due serie, stessi marker/colore per metodo)
    for m in methods_sorted:
        xi = overhead[m]
        ax.plot([xi], [acc_img[m]], linestyle="", marker=markers[m], markersize=ms,
                markerfacecolor=colors[m], markeredgecolor="black", markeredgewidth=0.9, zorder=3)
        ax.plot([xi], [acc_plc[m]], linestyle="", marker=markers[m], markersize=ms,
                markerfacecolor=colors[m], markeredgecolor="black", markeredgewidth=0.9, alpha=0.9, zorder=3)

    # linee di tendenza
    ax.plot(x, y_img, linestyle=line_styles["ImageNet-1K"], linewidth=lw, color=line_colors["ImageNet-1K"], zorder=2)
    ax.plot(x, y_plc, linestyle=line_styles["Places365"],  linewidth=lw, color=line_colors["Places365"],  zorder=2)

    # titolo minimale + assi
    ax.set_title(dataset_name, pad=8)
    ax.set_xlabel("Server-side overhead (relative units)", labelpad=6)
    ax.set_ylabel("Top-1 Accuracy (%)", labelpad=6)
    ax.set_xticks(x, [m.replace('_full','').replace('_','-') for m in methods_sorted])

    # annotazioni (valori)
    for xi, yi in zip(x, y_img):
        ax.text(xi, yi+0.9, f"{yi:.2f}", ha="center", va="bottom", fontsize=11, color="#1f1f1f")
    for xi, yi in zip(x, y_plc):
        ax.text(xi, yi-1.1, f"{yi:.2f}", ha="center", va="top", fontsize=11, color="#1f1f1f")

    # legende: metodi in basso a destra; pretraining in alto a destra appena sopra
    method_handles = [Line2D([0],[0], marker=markers[m], linestyle="", markersize=ms,
                             markerfacecolor=colors[m], markeredgecolor="black", markeredgewidth=0.9,
                             label=m.replace('_full','').replace('_','-')) for m in methods_sorted]
    pretrain_handles = [
        Line2D([0],[0], color=line_colors["ImageNet-1K"], lw=lw, linestyle=line_styles["ImageNet-1K"], label="ImageNet-1K pretrain"),
        Line2D([0],[0], color=line_colors["Places365"],  lw=lw, linestyle=line_styles["Places365"],  label="Places365 pretrain"),
    ]
    leg_methods = ax.legend(handles=method_handles, title="Methods", loc="lower right", ncol=3,
                            frameon=True, fancybox=True, framealpha=0.95, borderaxespad=0.6)
    ax.add_artist(leg_methods)
    # piazzata poco sopra la precedente
    leg_pretrain = ax.legend(handles=pretrain_handles, title="Pretraining",
                             loc="lower right", bbox_to_anchor=(1.0, 0.22),
                             frameon=True, fancybox=True, framealpha=0.95)
    ax.add_artist(leg_pretrain)

    stylize_axes(ax)
    plt.tight_layout()

    # --- Salva PDF se richiesto ---
    if save_pdf:
        os.makedirs(outdir, exist_ok=True)
        if slug is None:
            slug = dataset_name.lower().replace(' ', '_').replace('-', '')
        pdf_path = os.path.join(outdir, f"{slug}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")  # vettoriale
        print(f"Saved PDF -> {pdf_path}")

    plt.show()

# ------------------------ Esecuzione ------------------------
plot_acc_vs_overhead("CIFAR-10",   acc_imagenet_c10,   acc_places_c10,
                     save_pdf=True, outdir="./figures", slug="acc_overhead_cifar10")

plot_acc_vs_overhead("CIFAR-100-C", acc_imagenet_c100c, acc_places_c100c,
                     save_pdf=True, outdir="./figures", slug="acc_overhead_cifar100c")




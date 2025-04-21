# style.py
import matplotlib

plot_config = {
    "axes.prop_cycle": matplotlib.cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]),
    "axes.edgecolor": "#333",
    "axes.facecolor": "#fafafa",
    "axes.labelcolor": "#111",
    "figure.facecolor": "#fff",
    "xtick.color": "#111",
    "ytick.color": "#111",
    "grid.color": "#ccc"
}

matplotlib.rcParams.update(plot_config)

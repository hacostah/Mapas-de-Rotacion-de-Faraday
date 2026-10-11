"""Salida común de figuras y datos reproducibles del ejemplo."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[2] / "results" / "falso_observatorio"

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "figure.facecolor": "white",
        "pdf.fonttype": 42, "axes.grid": False,
    })

def save_figure(fig, name, output_dir=None):
    dest = RESULTS if output_dir is None else Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(dest / f"{name}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)

def write_json(path, data):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(data, default=convert, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")

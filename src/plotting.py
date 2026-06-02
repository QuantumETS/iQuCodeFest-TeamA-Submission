from pathlib import Path
import matplotlib.pyplot as plt


def save_fig(save_dir=Path("figures"), filename="figure.png", fig=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / f"{filename}"
    if fig is not None:
        fig.savefig(fig_path)
    else:
        plt.savefig(fig_path)

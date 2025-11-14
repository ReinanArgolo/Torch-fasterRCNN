import os
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_and_save(x_epochs, y_values, ylabel, out_file):
    plt.figure()
    plt.plot(x_epochs, y_values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def export_metrics(exp_dir, epochs, train_loss, val_loss, map_vals):
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "map": map_vals,
            },
            f,
            indent=2,
        )
    with open(os.path.join(exp_dir, "metrics.csv"), "w") as f:
        f.write("epoch,train_loss,val_loss,map\n")
        for e, tl, vl, mp in zip(epochs, train_loss, val_loss, map_vals):
            f.write(f"{e},{tl},{vl},{mp}\n")
    plot_and_save(epochs, train_loss, "Train Loss", os.path.join(exp_dir, "train_loss.png"))
    if any(not math.isnan(v) for v in val_loss):
        plot_and_save(epochs, val_loss, "Val Loss", os.path.join(exp_dir, "val_loss.png"))
    if any(not math.isnan(m) for m in map_vals):
        plot_and_save(epochs, map_vals, "mAP", os.path.join(exp_dir, "map.png"))
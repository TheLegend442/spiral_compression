
from transformer import SpiralTransformer, SpiralDataset, spiral_collate

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

## EVALUATION

@torch.no_grad()
def _compute_single_spiral_loss(
        out: dict,
        sample: dict,
        padding_mask: torch.Tensor,
        device,
        k_loss_fn=nn.SmoothL1Loss(), 
        w_k=0.5,
        w_k_tight=0.5,
        w_angle=0.7
):
    """Compute the same loss terms as `_run_eval`, but for a single (non-padded) sample.

    Returns:
        total_loss (float) or None if required GT fields are missing.
        breakdown (dict[str,float]) with per-term losses.
    """

    required = ["tight_onehot", "tightness", "normality", "k"]
    if not all(k in sample for k in required):
        return None, {}

    # Targets
    k_true = batch["k"].to(device).float()
    k_tight_true = batch["k_tight"].to(device).float()
    theta_tight_true = batch["theta_tight"].to(device).float()
    is_tight = batch["is_tight"].to(device).float()
    
    if k_true.ndim == 2 and k_true.shape[-1] == 1:
        k_true = k_true.squeeze(-1)
    if k_tight_true.ndim == 2 and k_tight_true.shape[-1] == 1:
        k_tight_true = k_tight_true.squeeze(-1)

    # Predictions
    k_pred              = out["k_pred"]
    k_tight_pred        = out["k_tight_pred"]
    ang_raw             = out["ang_raw"].view(-1)     # [B]
    
    loss_k          = k_loss_fn(k_pred, k_true)
    loss_k_tight    = k_loss_fn(k_tight_pred, k_tight_true)
    
    mask = is_tight > 0.5   # [B] bool
    theta = theta_tight_true.float().view(-1)         # [B]
    per = (ang_raw - theta) ** 2                      # [B]
    loss_angle = (per * mask).sum() / mask.sum().clamp(min=1.0)

    loss = (
        w_k         * loss_k       +
        w_k_tight   * loss_k_tight +
        w_angle     * loss_angle
    )

    breakdown = {
        "k": float(loss_k.item()),
        "k_tight": float(loss_k_tight.item()),
        "theta_tight": float(loss_angle.item()),
        "total": float(loss.item()),
    }
    return float(loss.item()), breakdown

def run_inference_and_plot(
        model,
        dataset: "SpiralDataset",
        device,
        thr_tight: float = 0.5,
        plot_title: str = "Spiral Prediction",
        plot_type: str = "tight",
        compute_loss: bool = True,
        k_loss_fn=nn.SmoothL1Loss(), 
        w_k=0.5,
        w_k_tight=0.5,
        w_angle=0.7,
        neigh_df=None
):
    """
    Runs model on a single spiral, maps patch-level predictions back to T,
    and calls `plot_tight` with predicted outputs.

    - tight/flat are multi-label per patch -> sigmoid -> threshold
    - severity/normality are logits -> sigmoid to [0,1]
    - k is one value per spiral
    """
    model.eval()
    spiral_preds = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]

            seq_emb = sample["seq_emb"].unsqueeze(0).to(device)  # [1, L, d_emb]
            L = sample["seq_emb"].shape[0]
            padding_mask = torch.zeros(1, L, dtype=torch.bool, device=device)

            out = model(seq_emb, padding_mask=padding_mask)

            # optional per-spiral loss (only if GT is present in the sample)
            loss_val = None
            loss_breakdown = {}
            if compute_loss:
                loss_val, loss_breakdown = _compute_single_spiral_loss(
                    out=out,
                    sample=sample,
                    padding_mask=padding_mask,
                    device=device,
                    k_loss_fn=k_loss_fn, 
                    w_k=w_k,
                    w_k_tight=w_k_tight,
                    w_angle=w_angle
                )


            k_pred = out["k_pred"][0].item()  # scalar
            k_tight_pred = out["k_tight_pred"][0].item()  # scalar
            ang_raw_pred = out["ang_raw"][0].item() # scalar


            theta = sample["theta"].cpu().numpy()  # [T]
            r     = sample["r"].cpu().numpy()      # [T]
            k     = sample["k"].item()             # scalar
            k_tight = sample["k_tight"].item()     # scalar
            theta_tight = sample.get("theta_tight", None) # scalar or None
        
            # build dict for plotting / analysis
            spiral_pred = {
                "theta": theta,
                "r": r,
                "k": k,
                "k_tight": k_tight,
                "theta_tight": theta_tight,

                # predictions
                "k_pred": k_pred,
                "k_tight_pred": k_tight_pred,
                "ang_raw_pred": ang_raw_pred % (2 * np.pi), # wrap to [0, 2pi)

                # loss (if computed)
                "loss": loss_val,
                "loss_breakdown": loss_breakdown,
            }

            spiral_preds.append(spiral_pred)
            
    num_spirals = len(spiral_preds)
    if num_spirals > 1:

        fig, axes = plt.subplots(num_spirals,2, figsize=(14, 6*num_spirals),
                                 gridspec_kw={'wspace': 0.2, 'hspace':0.4})
        for i in range(num_spirals):
            fig.delaxes(axes[i, 1])
            axes[i, 1] = fig.add_subplot(num_spirals, 2, 2*i + 2, projection="polar")
            plot_tight_distribution(spiral_preds[i], plot_title=f"", ax1=axes[i,0], ax2=axes[i,1], threshold=thr_tight)
            
            # plot_nearby_spirals(spiral_preds[i], neigh_df, plot_title="Nearby Spirals", num_neighbors=3, save_path=f"../results/nearby_spirals_plot{i}.png")
            
        plt.suptitle(plot_title)
        plt.savefig(f"../results/multiple_spiral_plots_{plot_type}_dist.png", dpi=300)
            
    else:
        title_dist = "Tightness Distribution"
        if compute_loss and spiral_preds[0].get("loss") is not None:
            title_dist = f"Tightness Distribution | loss={spiral_preds[0]['loss']:.4f}"
        elif compute_loss:
            title_dist = "Tightness Distribution | loss=N/A"

        plot_tight_distribution(spiral_preds[0], plot_title=title_dist)

def plot_tight_distribution(spiral_dict, plot_title, VIEW_WIDTH=5 * np.pi / 50, ax1=None, ax2=None, threshold=None):

    ## PARSE THE SPIRAL
    theta = np.asarray(spiral_dict['theta'])
    r = np.asarray(spiral_dict['r'])
    
    GROUND_TRUTH = False
    if 'k' in spiral_dict and spiral_dict['k'] is not None:
        k = spiral_dict['k']
        k_tight = spiral_dict['k_tight']
        # tightness_ang = np.asarray(spiral_dict['theta_tight'])
        GROUND_TRUTH = True
    
    k_pred = spiral_dict['k_pred']
    k_tight_pred = spiral_dict['k_tight_pred']
    theta_tight_pred = np.asarray(spiral_dict['ang_raw_pred'])
    
    IS_TIGHT = True
    if threshold is not None:
        IS_TIGHT = (k_tight_pred / k_pred) < threshold
    
    ## DIVERGENCE CALCULATION
    def get_divergence(_theta, _r, view_angle, view_width):
        r_tight = []
        r_normal = []
        theta_tight = []
        theta_normal = []
        
        tight_slots = []
        ang = view_angle % (2*np.pi)

        while ang < _theta[-1] + view_width / 2:
            tight_slots.append((ang - view_width/2, ang + view_width/2))
            ang += 2*np.pi
            
        normal_slots = []
        ang = view_angle + np.pi
        while ang < _theta[-1] + view_width / 2:
            normal_slots.append((ang - view_width/2, ang + view_width/2))
            ang += 2*np.pi
            
        for ts in tight_slots:
            idxs = np.where((_theta >= ts[0]) & (_theta <= ts[1]))[0]
            r_tight.extend(_r[idxs])
            theta_tight.extend(_theta[idxs])
            
        for ns in normal_slots:
            idxs = np.where((_theta >= ns[0]) & (_theta <= ns[1]))[0]
            r_normal.extend(_r[idxs])
            theta_normal.extend(_theta[idxs])
            
        return np.array(r_tight), np.array(r_normal), np.array(theta_tight), np.array(theta_normal)
    
    r_tight, r_normal, theta_tight, theta_normal = get_divergence(theta, r, theta_tight_pred, VIEW_WIDTH)
    
    if np.mean(r_tight) > np.mean(r_normal):
        r_tight, r_normal = r_normal, r_tight
        theta_tight, theta_normal = theta_normal, theta_tight
        theta_tight_pred += np.pi
    
    ## PLOTTING VALUES
    # Gaussian Fit
    tight_side_sigma = np.std(r_tight)
    tight_side_mean = np.mean(r_tight)
    normal_side_sigma = np.std(r_normal)
    normal_side_mean = np.mean(r_normal)
    
    # Angles
    mid_theta = theta_tight_pred
    lb_theta = mid_theta - VIEW_WIDTH / 2
    ub_theta = mid_theta + VIEW_WIDTH / 2
    
    # New r
    r_max = max(r) * 1.1
    r_plot = np.linspace(0, r_max, 200)
    
    # Figure
    PLOT_FIGURE = False
    if ax1 is None or ax2 is None:    
        PLOT_FIGURE = True
        fig, (ax1, ax_dummy) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={'wspace': 0.2}
        )
        # Replace second axes with polar
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        ax_dummy.remove()
        
    ax_hist = ax1
    ax_polar = ax2
    
    ## POLAR PLOT
    ax_polar.plot(theta, r, linewidth=2, color='gray', alpha=0.7)
    
    def draw_angle_band(ax, theta_mid, theta_lb, theta_ub, r, region_color):

        # bounds
        ax.plot(np.full_like(r, theta_lb), r, linestyle="--", color=region_color, alpha=0.6)
        ax.plot(np.full_like(r, theta_ub), r, linestyle="--", color=region_color, alpha=0.6)

        # ---- shaded band ----
        theta = np.concatenate([
            np.full_like(r, theta_lb),
            np.full_like(r, theta_ub)[::-1],
        ])
        r_fill = np.concatenate([r, r[::-1]])

        ax.fill(theta, r_fill, color=region_color, alpha=0.25, linewidth=0)
        
        # central angle
        ax.plot(np.full_like(r, theta_mid), r, linestyle="-", color="navy", alpha=0.75)
    
    if IS_TIGHT:
        # Tigth side
        draw_angle_band(ax_polar, mid_theta, lb_theta, ub_theta, r_plot, region_color='orange')
        # Normal side
        draw_angle_band(ax_polar, mid_theta + np.pi, lb_theta + np.pi, ub_theta + np.pi, r_plot, region_color='forestgreen')
    
        ax_polar.scatter(theta_tight, r_tight, color='darkorange', s=20, zorder=3, alpha=0.9)
        ax_polar.scatter(theta_normal, r_normal, color='darkgreen', s=20, zorder=3, alpha=0.9)
    
    ax_polar.set_rmax(r_max)
    ax_polar.grid(alpha=0.3)
    if not IS_TIGHT: ax_polar.set_title("IZRAZITA STISNJENOST NI ZAZNANA\n", fontsize=14, color='red')
    
    if IS_TIGHT:    
        if theta_tight_pred < 0:
            theta_tight_pred += 2 * np.pi
        ha = 'left' if (np.pi/2 > theta_tight_pred or theta_tight_pred > 3*np.pi/2) else 'right'

        ax_polar.text(theta_tight_pred, r_max*1.05, f"{theta_tight_pred:.2f}π", color='navy', ha=ha, va='center', fontsize=13)
    
    ax_polar.set_yticklabels([])
    
    # ## HISTOGRAM + GAUSSIANS
    bins = np.linspace(0, r_max, 30)

    def gauss_func(x, mu, sigma):
        coeff = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)
    
    
    bw = (bins[1] - bins[0])

    # --- Tight (right) ---
    x_tight = bins
    gauss_tight = gauss_func(x_tight, tight_side_mean, tight_side_sigma)
    y_tight = gauss_tight * len(r_tight) * bw

    ax_hist.plot(x_tight, y_tight, color="orange", linestyle="--", label="Gaussova distribucija stisnjene strani")
    ax_hist.fill_between(x_tight, y_tight, color="orange", alpha=0.15)

    # --- Normal (left) ---
    # mirror x to negative AND keep x increasing for fill_between
    x_normal = -bins[::-1]  # goes from -max .. 0 (increasing)
    mu_normal_left = -normal_side_mean  # mirror the mean to the left
    gauss_normal = gauss_func(x_normal, mu_normal_left, normal_side_sigma)
    y_normal = gauss_normal * len(r_normal) * bw

    ax_hist.plot(x_normal, y_normal, color="forestgreen", linestyle="--", label="Gaussova distribucija običajne strani")
    ax_hist.fill_between(x_normal, y_normal, color="forestgreen", alpha=0.15)

    ax_hist.axvline(0.0, color="black", linewidth=1, alpha=0.5)
    ax_hist.set_xlim(-bins.max(), bins.max())  # avoid any autoscale weirdness
    ax_hist.set_ylim(0, np.max([y_tight.max(), y_normal.max()]) * 1.4)
    ax_hist.set_xlabel("Polmer")
    ax_hist.set_ylabel("Število")
    ax_hist.legend(loc="upper left", fontsize=10)
    ax_hist.grid(alpha=0.3)
    
    ## DISTRIBUTION SIMILARITY
    def hellinger_gaussians(mu1, sigma1, mu2, sigma2, eps=1e-12):
        s1 = max(float(sigma1), eps)
        s2 = max(float(sigma2), eps)

        denom = s1*s1 + s2*s2
        coeff = np.sqrt((2.0*s1*s2) / denom)
        expo = np.exp(-((mu1 - mu2)**2) / (4.0*denom))

        h2 = 1.0 - coeff * expo
        # numerical safety
        h2 = float(np.clip(h2, 0.0, 1.0))
        return np.sqrt(h2)  # in [0,1]
    
    hell = hellinger_gaussians(tight_side_mean, tight_side_sigma, normal_side_mean, normal_side_sigma)
    
    # Set title to histogram
    
    # 1) main header
    if GROUND_TRUTH:
        ax_hist.set_title(f"Generirani koeficient k/k_tight = {k/k_tight:.3f} | napovedan = {k_pred/k_tight_pred:.3f}", pad=4)
    else:
        ax_hist.set_title(f"Napovedani koeficient k/k_tight = {k_pred/k_tight_pred:.3f}", pad=4)

    min_bin = bins[0]
    
    # 2) Overlay colored lines in the reserved space
    ax_hist.text(
        0.02, 0.79,
        f"Stisnjena: μ={tight_side_mean:.1f}, σ={tight_side_sigma:.1f}",
        transform=ax_hist.transAxes,
        ha="left", va="bottom",
        color="orange",
        clip_on=False,
        fontsize=11,
    )

    ax_hist.text(
        0.02, 0.745,
        f"Običajna: μ={normal_side_mean:.1f}, σ={normal_side_sigma:.1f}",
        transform=ax_hist.transAxes,
        ha="left", va="bottom",
        color="forestgreen",
        clip_on=False,
        fontsize=11,
    )

    ax_hist.text(
        0.02, 0.70,
        f"Hellingerjeva razdalja: {hell:.3f}",
        transform=ax_hist.transAxes,
        ha="left", va="bottom",
        color="black",
        clip_on=False,
        fontsize=11,
    )
    
    ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
    if PLOT_FIGURE:
        plt.suptitle(plot_title)
        save_path= "../results/spiral_tight_divg_plot.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

def plot_nearby_spirals(spiral_dict, spiral_neighbors, plot_title="Nearby Spirals", num_neighbors=3, save_path=None):
    
    k_pred = spiral_dict['k_pred']
    k_tight_pred = spiral_dict['k_tight_pred']
    theta_tight_pred = np.asarray(spiral_dict['ang_raw_pred'])
    
    k_coeff = k_tight_pred / k_pred
    
    def find_neighbors(_k_coeff, _theta, neighbors, num_neighbors):
        
        neighbor_dists = []
        for i in range(len(neighbors)):
            n = neighbors.iloc[i]
            if 'tight' not in n['file_path']:
                continue
            coeff_diff = (n['k_coeff'] - _k_coeff) ** 2
            theta_diff = (n['theta_tight'] - _theta) ** 2
            neighbor_dists.append((coeff_diff + theta_diff, n))
            
        neighbor_dists.sort(key=lambda x: x[0])
        return [nd[1] for nd in neighbor_dists[:num_neighbors]]
    
    neighs = find_neighbors(k_coeff, theta_tight_pred, spiral_neighbors, num_neighbors)

    spirals2plot = [spiral_dict] + [np.load(n['file_path'], allow_pickle=True) for n in neighs]
    
    num_spirals = len(spirals2plot)
    rows = num_spirals // 2 + (num_spirals % 2)

    fig, axes = plt.subplots(
        rows, 2,
        subplot_kw={"projection": "polar"},
        figsize=(14, 6 * rows)
    )

    # Handle case when rows == 1 (axes is 1D)
    axes = np.atleast_2d(axes)

    for i in range(num_spirals):
        ax = axes[i // 2, i % 2]

        ax.plot(
            spirals2plot[i]["theta"],
            spirals2plot[i]["r"],
            linewidth=2,
            color="gray",
            alpha=0.7,
        )

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(False)

        if i == 0:
            ax.set_title(f"Input Spiral\nPredicted k_coeff = {spirals2plot[i]['k_pred']/spirals2plot[i]['k_tight_pred']:.3f} | Predicted theta: {spirals2plot[i]['ang_raw_pred']:.3f}", fontsize=16)
        else:
            ang = spirals2plot[i]['theta_tight']
            if ang is None:
                ang = "no data"
            ax.set_title(f"Neighbor {i}\nK_coeff = {spirals2plot[i]['k']/spirals2plot[i]['k_tight']:.3f} | Theta: {ang:.3f}", fontsize=16)

    # Hide unused subplots
    for j in range(num_spirals, rows * 2):
        axes[j // 2, j % 2].axis("off")

    plt.tight_layout()
    if save_path is None:
        save_path= "../results/nearby_spirals_plot.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def main():
    
    MODEL_PATH = "../models/" + "sp_trans_20260123_014021.pt"
    
    parser = argparse.ArgumentParser(
        description="Evaluate Spiral Transformer and plot results.",
    )
    
    parser.add_argument(
        "--spiral_path", type=str,
        help="Input spiral path for inference.",
        default="",
    )  
    
    test_spirals = [
        "../data/test_set/normal/normal_992.npz",
        "../data/test_set/spiky/spiky_8.npz",
        "../data/test_set/flat/flat_14.npz",
        "../data/test_set/tight/tight_44.npz",
        "../data/test_set/tight/tight_46.npz",
        "../data/test_set/tight/tight_50.npz",
    ]
    
    test_spirals = [Path(sp) for sp in test_spirals]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # neigh_df = pd.read_csv("./tight_spirals_tight_angles2.csv")
    
    args = parser.parse_args()
    input_path = Path(args.spiral_path)
    
    if input_path == Path(""):
        spirals = test_spirals
        dataset = SpiralDataset([spirals])

        sample = dataset[0]
        d_emb = sample["seq_emb"].shape[1]
        
        model = SpiralTransformer(
            embed_dim=d_emb,
            d_model=256,
            num_heads=8,
            num_layers=3,
            d_ff=512,
            dropout=0.1,
            max_seq_len=1024,
        ).to(device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        run_inference_and_plot(
            model=model,
            dataset=dataset,
            device=device,
            thr_tight=0.5,
            plot_title="Spiral Tightening Prediction",
            plot_type="tight",
        )
    else:
        dataset = SpiralDataset([[input_path]])
        
        sample = dataset[0]
        d_emb = sample["seq_emb"].shape[1]

        model = SpiralTransformer(
            embed_dim=d_emb,
            d_model=256,
            num_heads=8,
            num_layers=3,
            d_ff=512,
            dropout=0.1,
            max_seq_len=1024,
        ).to(device)
        
        # Load the pretrained model weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        run_inference_and_plot(
            model=model,
            dataset=dataset,
            device=device,
            thr_tight=0.5,
            plot_title="Spiral Tightening Prediction",
            plot_type="tight"
        )
        
    
    ## EVAULATION CODE
    # model = SpiralTransformer(
    #     embed_dim=d_emb,
    #     d_model=256,
    #     num_heads=8,
    #     num_layers=3,
    #     d_ff=512,
    #     dropout=0.1,
    #     max_seq_len=1024,
    # ).to(device)
    
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # model.eval()
    
    # data_dirs = [Path("../data/test_set/normal"), Path("../data/test_set/tight"), Path("../data/test_set/flat"), Path("../data/test_set/spiky")]
    # spiral_paths = []
    # for d in data_dirs:
    #     files = sorted(d.glob("*.npz"))
    #     spiral_paths.append(files)
    # eval_dataset = SpiralDataset(spiral_paths)
    # print(f"Evaluation dataset size: {len(eval_dataset)} spirals.")
    # eval_loader = DataLoader(eval_dataset, batch_size=16, collate_fn=spiral_collate)
    
    # eval_loss = _run_eval(
    #     model=model,
    #     loader=eval_loader,
    #     device=device,
    #     k_loss_fn=nn.SmoothL1Loss(),
    #     w_k=0.5,
    #     w_k_tight=0.5,
    #     w_angle=0.7
    # )
    
    # print(f"Evaluation Loss on Test Set: {eval_loss:.4f}")
    
if __name__ == "__main__":
    main()
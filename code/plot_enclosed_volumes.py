# plot_enclosed_volumes.py

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise error column
    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("").astype(str)

    # Coerce numeric columns
    for c in ["vol_vox_m3", "voxel_size", "enclosed_voxels", "blocked_voxels", "y_bottom", "y_top"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def read_timeseries(path: str) -> pd.DataFrame:
    ts = pd.read_csv(path)
    for c in ["step", "t", "frame", "substep", "vol_vox", "enclosed_voxels", "blocked_voxels", "y_bottom", "y_top"]:
        if c in ts.columns:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")
    ts["object"] = ts["object"].astype(str)
    ts = ts.dropna(subset=["object", "frame", "vol_vox"])
    return ts


def plot_summary(df: pd.DataFrame, out_dir: str, topk: int | None = None) -> None:
    ok = df[(df["error"] == "") | (df["error"].str.lower() == "nan")].copy()
    ok = ok.dropna(subset=["vol_vox_m3", "voxel_size", "enclosed_voxels"])

    ok["vol_check"] = ok["enclosed_voxels"] * (ok["voxel_size"] ** 3)
    ok["vol_abs_err"] = (ok["vol_vox_m3"] - ok["vol_check"]).abs()

    ok = ok.sort_values("vol_vox_m3", ascending=False)
    if topk is not None and topk > 0:
        ok = ok.head(topk)

    os.makedirs(out_dir, exist_ok=True)

    # Bar plot (sorted)
    fig_h = max(4.0, 0.35 * len(ok))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(ok["object"], ok["vol_vox_m3"])
    ax.invert_yaxis()
    ax.set_xlabel("Enclosed volume (m^3)")
    ax.set_title("Enclosed volumes per object (final frame)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_enclosed_volume_bar.png"), dpi=200)
    plt.close(fig)

    # Sanity check scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ok["vol_check"], ok["vol_vox_m3"])
    ax.set_xlabel("enclosed_voxels * voxel_size^3 (m^3)")
    ax.set_ylabel("vol_vox_m3 (m^3)")
    ax.set_title("Sanity check (should lie on y=x)")
    # Make axes comparable
    mn = float(np.nanmin([ok["vol_check"].min(), ok["vol_vox_m3"].min()]))
    mx = float(np.nanmax([ok["vol_check"].max(), ok["vol_vox_m3"].max()]))
    ax.plot([mn, mx], [mn, mx])
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_sanity_scatter.png"), dpi=200)
    plt.close(fig)

    # Save a sorted table too
    ok_out = ok[["object", "vol_vox_m3", "voxel_size", "enclosed_voxels", "blocked_voxels", "y_bottom", "y_top", "shape", "vol_abs_err"]].copy()
    ok_out.to_csv(os.path.join(out_dir, "summary_sorted.csv"), index=False)


def plot_timeseries(ts: pd.DataFrame, out_dir: str, stride: int = 10, objects: list[str] | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    data = ts.copy()
    if objects:
        data = data[data["object"].isin(objects)].copy()

    data = data.sort_values(["object", "frame"])

    if stride is not None and stride > 1:
        data = data[data["frame"] % stride == 0].copy()

    # Overlay per object (no legend by default, too many lines)
    fig, ax = plt.subplots(figsize=(12, 5))
    for obj, g in data.groupby("object", sort=True):
        ax.plot(g["frame"].to_numpy(), g["vol_vox"].to_numpy(), alpha=0.35)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Enclosed volume (m^3)")
    ax.set_title("Enclosed volume over time (each line is one object)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timeseries_overlay.png"), dpi=200)
    plt.close(fig)

    # Mean ± std across objects per frame (only meaningful if many objects)
    agg = data.groupby("frame", as_index=False)["vol_vox"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(agg["frame"].to_numpy(), agg["mean"].to_numpy())
    m = agg["mean"].to_numpy()
    s = agg["std"].to_numpy()
    ax.fill_between(agg["frame"].to_numpy(), m - s, m + s, alpha=0.2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Enclosed volume (m^3)")
    ax.set_title("Enclosed volume over time (mean ± std across objects)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timeseries_mean_std.png"), dpi=200)
    plt.close(fig)

    # Optional: individual plots per object (only if user passed a filtered object list)
    if objects and len(objects) > 0:
        per_dir = os.path.join(out_dir, "per_object")
        os.makedirs(per_dir, exist_ok=True)
        for obj, g in data.groupby("object", sort=True):
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(g["frame"].to_numpy(), g["vol_vox"].to_numpy())
            ax.set_xlabel("Frame")
            ax.set_ylabel("Enclosed volume (m^3)")
            ax.set_title(f"Enclosed volume over time: {obj}")
            fig.tight_layout()
            fig.savefig(os.path.join(per_dir, f"{obj}.png"), dpi=200)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="logs/ycb_voxel_volume_sweep.csv")
    ap.add_argument("--ts", default="logs/ycb_voxel_volume_timeseries_all.csv")
    ap.add_argument("--out", default="plots_enclosed_volume")
    ap.add_argument("--stride", type=int, default=10, help="Downsample timeseries by frame modulo stride")
    ap.add_argument("--topk", type=int, default=0, help="If >0, only plot top K objects by final volume")
    ap.add_argument("--objects", nargs="*", default=None, help="If set, only plot these objects (also makes per object plots)")
    args = ap.parse_args()

    df = read_summary(args.summary)
    ts = read_timeseries(args.ts)

    topk = args.topk if args.topk and args.topk > 0 else None

    plot_summary(df, args.out, topk=topk)
    plot_timeseries(ts, args.out, stride=args.stride, objects=args.objects)

    print(f"Saved plots to: {args.out}")


if __name__ == "__main__":
    main()

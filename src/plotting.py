import numpy as np

def _require(diagnostic_data, key):
    fn = diagnostic_data.get(key)
    if fn is None:
        raise ValueError(f"diagnostic_data missing required entry '{key}'")
    return fn

def nn_training_diagnostics_plot(diagnostic_data):
    import matplotlib.pyplot as plt
    batched_model = _require(diagnostic_data, "batched_model_fn")
    unwhiten = _require(diagnostic_data, "unwhiten_fn")
    lossvis = _require(diagnostic_data, "lossvis_fn")

    def euclidean_to_lat_long(vectors):
        vecs = np.asarray(vectors)
        r = np.linalg.norm(vecs[:, :3], axis=-1)
        lat = np.arcsin(np.clip(vecs[:, 2] / np.maximum(r, 1e-12), -1.0, 1.0))
        lon = np.arctan2(vecs[:, 1], vecs[:, 0])
        return lat, lon

    params = diagnostic_data["parameters"]
    features = diagnostic_data["features"]
    forces = diagnostic_data["forces"]
    feature_stats = diagnostic_data["feature_stats"]
    target_stats = diagnostic_data["target_stats"]
    sun_grid = diagnostic_data.get("sun_grid")
    component_labels = diagnostic_data.get("component_labels", ("X", "Y", "Z"))

    pred_pm1_full = batched_model(features, params)
    pred01_full = 0.5 * (pred_pm1_full + 1.0)
    force01_full = 0.5 * (forces + 1.0)
    pred_real_full = unwhiten(pred01_full, target_stats)
    force_real_full = unwhiten(force01_full, target_stats)

    diff = np.abs(np.array(pred_real_full) - np.array(force_real_full))
    sum_abs = np.abs(np.array(pred_real_full)) + np.abs(np.array(force_real_full))
    smape_map = 200.0 * diff / np.maximum(sum_abs, 1e-12)

    reshaped = smape_map
    if sun_grid:
        try:
            reshaped = smape_map.reshape(*sun_grid, smape_map.shape[-1])
        except Exception:
            pass
    clipped = np.clip(reshaped, 0.0, 200.0)

    num_components = clipped.shape[-1] if clipped.ndim >= 3 else 1
    fig_map, axes_map = plt.subplots(1, num_components, figsize=(5 * num_components, 4), sharex=True)
    if num_components == 1:
        axes = [axes_map]
    else:
        axes = axes_map
    extent = (-180.0, 180.0, -90.0, 90.0)
    for comp_idx, ax in enumerate(axes):
        slice_data = clipped[..., comp_idx] if clipped.ndim >= 3 else clipped
        im = ax.imshow(slice_data, origin='lower', extent=extent, aspect='auto',
                       cmap='nipy_spectral', vmin=0.0, vmax=200.0)
        ax.set_ylabel('Latitude [deg]')
        label = component_labels[comp_idx] if comp_idx < len(component_labels) else f"{comp_idx}"
        ax.set_title(f'SMAPE Δ{label}')
        fig_map.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='SMAPE [%]')
    axes[-1].set_xlabel('Longitude [deg]')
    fig_map.tight_layout()
    plt.show()

    train_features = diagnostic_data["train_features"]
    train_forces = diagnostic_data["train_forces"]
    train_feat_real = unwhiten(train_features, feature_stats)
    lat, lon = euclidean_to_lat_long(train_feat_real)
    pred_pm1 = batched_model(train_features, params)
    pred01 = 0.5 * (pred_pm1 + 1.0)
    force01 = 0.5 * (train_forces + 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    def colorize(arr):
        a = np.asarray(arr)
        if a.ndim > 1 and a.shape[-1] not in (3, 4):
            return np.linalg.norm(a, axis=-1)
        return a

    axes[0].scatter(np.degrees(lon), np.degrees(lat), c=colorize(force01), s=1)
    axes[0].set_title("GT proxy (scaled)")
    axes[1].scatter(np.degrees(lon), np.degrees(lat), c=colorize(pred01), s=1)
    axes[1].set_title("Pred proxy (scaled)")

    train_losses = np.array(diagnostic_data.get('train_losses', []), dtype=np.float64)
    val_losses = np.array(diagnostic_data.get('val_losses', []), dtype=np.float64)
    loss_steps = np.array(diagnostic_data.get('loss_steps', []), dtype=np.float64)
    axes[2].set_title('Smoothed loss')
    axes[2].set_yscale('log')
    train_sm = lossvis(train_losses)
    val_sm = lossvis(val_losses)

    def plot_series(smoothed, raw_losses, label, style='-'):
        use_smoothed = smoothed.size >= 2
        series = np.array(smoothed if use_smoothed else raw_losses, dtype=np.float64)
        if series.size == 0:
            return
        if loss_steps.size:
            xs = loss_steps[-series.size:] if loss_steps.size >= series.size else loss_steps
            if xs.size != series.size:
                min_len = min(xs.size, series.size)
                xs = xs[-min_len:]
                series = series[-min_len:]
        else:
            xs = np.arange(series.size, dtype=np.float64)
        markers = dict(marker='o', markersize=3) if series.size == 1 else {}
        axes[2].plot(xs, series, style, label=label, **markers)

    plot_series(train_sm, train_losses, 'train')
    plot_series(val_sm, val_losses, 'val', style='--')

    axes[2].legend()
    axes[2].set_xlabel('Step')
    plt.tight_layout()
    plt.show()

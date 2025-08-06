import os
import matplotlib.pyplot as plt


def parse_train_file(file_path):
    times = []
    avg_losses = []
    pix_losses = []
    latent_losses = []
    lr_indices = []

    with open(file_path) as f:
        lines = f.readlines()
    has_pix = any("pix_loss:" in line for line in lines)
    has_latent = any("latent_loss:" in line for line in lines)

    step = 0
    for line in lines:

        if "avg_loss:" in line:
            times.append(step)
            parts = line.strip().split(",")
            # avg_loss
            avg_losses.append(float(parts[0].split("avg_loss:")[1]))
            # pix_loss
            if has_pix:
                pix_part = [p for p in parts if "pix_loss:" in p][0]
                pix_losses.append(float(pix_part.split("pix_loss:")[1]))
            # latent_loss
            if has_latent:
                lat_part = [p for p in parts if "latent_loss:" in p][0]
                latent_losses.append(float(lat_part.split("latent_loss:")[1]))
            step += 1

        elif "Learning rate reduced to" in line:
            lr_indices.append(step)

    if not has_pix:
        pix_losses = None
    if not has_latent:
        latent_losses = None

    return times, avg_losses, pix_losses, latent_losses, lr_indices


def parse_val_file(file_path):
    times, losses = [], []
    with open(file_path) as f:
        idx = 0
        for line in f:
            if "avg_val_loss:" in line:
                times.append(idx)
                losses.append(float(line.split("avg_val_loss:")[1]))
                idx += 1
    return times, losses


def main():
    dir_path = "/home/wrz/src/python/SegRefiner/MyRefiner-VAE/output/25-07-31_15-31-51"
    train_file = os.path.join(dir_path, 'trainLoss.txt')
    val_file = os.path.join(dir_path, 'valLoss.txt')
    max_ticks = 20

    (t_steps, avg_losses, pix_losses,
     latent_losses, lr_indices) = parse_train_file(train_file)
    v_steps, val_losses = parse_val_file(val_file)

    if pix_losses is not None and latent_losses is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()

    step_train = max(1, len(t_steps) // max_ticks)
    xticks_train = t_steps[::step_train]

    # （1）avg_loss
    ax = axes[0]
    ax.plot(t_steps, avg_losses, '-o', label='avg_loss', markersize=4)

    first = True
    for idx in lr_indices:
        if first:
            ax.axvline(idx, linestyle='--', color='r', label='LR Reduced')
            first = False
        else:
            ax.axvline(idx, linestyle='--', color='r')
    ax.set_xticks(xticks_train)
    ax.set_xticklabels(xticks_train, rotation=45)
    ax.set_ylabel('avg_loss')
    ax.set_title('Train: avg_loss')
    ax.legend()

    if pix_losses is not None:
        ax = axes[1]
        ax.plot(t_steps, pix_losses, '-o', label='pix_loss', markersize=4)
        # 同样标注 LR
        for idx in lr_indices:
            ax.axvline(idx, linestyle='--', color='r')
        ax.set_xticks(xticks_train)
        ax.set_xticklabels(xticks_train, rotation=45)
        ax.set_ylabel('pix_loss')
        ax.set_title('Train: pix_loss')
        ax.legend()

    if latent_losses is not None:
        pos = 2 if pix_losses is not None else 1
        ax = axes[pos]
        ax.plot(t_steps, latent_losses, '-o', label='latent_loss', markersize=4)
        for idx in lr_indices:
            ax.axvline(idx, linestyle='--', color='r')
        ax.set_xticks(xticks_train)
        ax.set_xticklabels(xticks_train, rotation=45)
        ax.set_ylabel('latent_loss')
        ax.set_title('Train: latent_loss')
        ax.legend()

    pos_val = 3 if pix_losses is not None and latent_losses is not None else 1
    ax = axes[pos_val]
    ax.plot(v_steps, val_losses, '-o', label='val_loss', markersize=4)
    step_val = max(1, len(v_steps) // max_ticks)
    xticks_val = v_steps[::step_val]
    ax.set_xticks(xticks_val)
    ax.set_xticklabels(xticks_val, rotation=45)
    ax.set_xlabel('Index')
    ax.set_ylabel('avg_val_loss')
    ax.set_title('Validation Loss')
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

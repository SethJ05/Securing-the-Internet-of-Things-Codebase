
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
DEFAULT_CSV = Path(r"C:\Users\Joel_\Downloads\cleaned_dataset.csv")
CSV_PATH = Path(args[0]) if args else (DEFAULT_CSV if DEFAULT_CSV.exists() else HERE / "cleaned_dataset.csv")

LABEL_COL = "recording"      
SESSION_COL = "activity"     
RAW_FEATURES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
ADD_MAGNITUDE = True         
FEATURES = RAW_FEATURES + (["acc_mag", "gyro_mag"] if ADD_MAGNITUDE else [])

FS = 100           
WINDOW_SEC = 2.0
OVERLAP = 0.5
TRIM_SEC = 1.0      

CROSS_VALIDATE = True   
CROSS_VALIDATE = CROSS_VALIDATE or "--cv" in sys.argv
N_FOLDS = 5

EPOCHS = 80
BATCH = 64
LR = 5e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 15      
SEED = 42
OUT_DIR = HERE / "cnn_output_v2"


rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(path):
    df = pd.read_csv(path)
    needed = RAW_FEATURES + [LABEL_COL, SESSION_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}\nFound: {list(df.columns)}")
    df = df.dropna(subset=needed).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[SESSION_COL] = df[SESSION_COL].astype(str).str.strip()
    if ADD_MAGNITUDE:
        df["acc_mag"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
        df["gyro_mag"] = np.sqrt(df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)
    return df


def describe_sessions(df):
    info = (df.groupby(SESSION_COL, sort=False)
              .agg(label=(LABEL_COL, "nunique"), cls=(LABEL_COL, "first"), rows=(FEATURES[0], "size")))
    bad = info[info["label"] > 1]
    if len(bad):
        sys.exit("These sessions contain more than one class label: "
                 f"column is not a clean session id:\n{bad}")
    info["seconds_at_FS"] = (info["rows"] / FS).round(1)
    print("\nSessions per class:")
    print(info.groupby("cls").agg(sessions=("rows", "size"), rows=("rows", "sum"),
                                  median_seconds=("seconds_at_FS", "median")))
    return info


def make_windows(df, win, step, trim):
    X, y, g = [], [], []
    for sess, d in df.groupby(SESSION_COL, sort=False):   
        arr = d[FEATURES].to_numpy(np.float32)
        if len(arr) > 2 * trim + win:
            arr = arr[trim:len(arr) - trim]
        if len(arr) < win:
            print(f"  skipping {sess}: only {len(arr)} rows")
            continue
        label = d[LABEL_COL].iloc[0]
        for s in range(0, len(arr) - win + 1, step):
            X.append(arr[s:s + win])
            y.append(label)
            g.append(sess)
    return np.stack(X), np.array(y), np.array(g)


def carve_val(rest_info):
    val = []
    for cls, grp in rest_info.groupby("cls"):
        s = list(grp.index)
        rng.shuffle(s)
        n_val = max(1, round(0.15 * len(s))) if len(s) >= 3 else 0
        val += s[:n_val]
    val_set = set(val)
    train = [s for s in rest_info.index if s not in val_set]
    return train, val


def split_sessions(info):
    test = []
    for cls, grp in info.groupby("cls"):
        s = list(grp.index)
        rng.shuffle(s)
        if len(s) < 3:
        test += s[:max(1, round(0.2 * len(s))) if len(s) >= 2 else 0]
    test_set = set(test)
    train, val = carve_val(info.loc[[s for s in info.index if s not in test_set]])
    return {"train": train, "val": val, "test": test}


def cv_splits(info):
    sess = np.array(info.index)
    labels = info["cls"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = []
    for tr_idx, te_idx in sgkf.split(sess, labels, groups=sess):
        train, val = carve_val(info.loc[sess[tr_idx]])
        out.append({"train": train, "val": val, "test": list(sess[te_idx])})
    return out


class WinDS(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)          # (N, C, T)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = x * (1 + 0.15 * torch.randn(x.shape[0], 1))       # per-channel scaling
            x = x + 0.1 * torch.randn_like(x)                     # noise
            shift = int(torch.randint(-x.shape[1] // 10, x.shape[1] // 10 + 1, (1,)))
            x = torch.roll(x, shift, dims=1)                      # small time shift
        return x, self.y[i]


class CNN1D(nn.Module):
    def __init__(self, channels, n_classes):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

        self.features = nn.Sequential(
            block(channels, 16, 5), nn.MaxPool1d(2), nn.Dropout(0.2),
            block(16, 32, 5), nn.MaxPool1d(2), nn.Dropout(0.2),
            block(32, 64, 3), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


def predict(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def fit(Xtr, ytr, Xva, yva, n_classes, verbose=True):
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=device)
    tr_loader = DataLoader(WinDS(Xtr, ytr, augment=True), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(WinDS(Xva, yva), batch_size=BATCH)

    model = CNN1D(Xtr.shape[1], n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": [], "va_f1": []}
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        trl, tra = run_epoch(model, tr_loader, loss_fn, opt)
        val, vaa = run_epoch(model, va_loader, loss_fn)
        vaf1 = f1_score(yva, predict(model, Xva), average="macro", zero_division=0)
        for k, v in zip(hist, (trl, val, tra, vaa, vaf1)):
            hist[k].append(v)
        if verbose:
            print(f"epoch {ep:3d}  train loss {trl:.3f} acc {tra:.3f} | "
                  f"val loss {val:.3f} acc {vaa:.3f} macro-F1 {vaf1:.3f}")
        if vaf1 > best + 1e-4:
            best, best_ep, bad = vaf1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                if verbose:
                    print("early stopping")
                break
    model.load_state_dict(best_state)
    return model, hist, best_ep


def run_split(X, y, groups, split, n_classes, verbose=True):
    idx = {k: np.isin(groups, v) for k, v in split.items()}
    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xva, yva = X[idx["val"]], y[idx["val"]]
    Xte, yte = X[idx["test"]], y[idx["test"]]
    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        sys.exit("you need more sessions per class.")


    mean = Xtr.reshape(-1, Xtr.shape[2]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-8
    prep = lambda a: np.ascontiguousarray(((a - mean) / std).transpose(0, 2, 1), dtype=np.float32)

    if verbose:
        counts = np.bincount(ytr, minlength=n_classes)
        print("Train windows per class:", counts.tolist())
    model, hist, best_ep = fit(prep(Xtr), ytr, prep(Xva), yva, n_classes, verbose)
    pred = predict(model, prep(Xte))
    return model, hist, best_ep, yte, pred, mean, std


def plot_results(hist, cm, classes, path):
    labels = list(range(len(classes)))
    ncols = 3 if hist else 1
    fig, ax = plt.subplots(1, ncols, figsize=(16 if hist else 6, 4.5))
    ax = np.atleast_1d(ax)
    if hist:
        ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(hist["tr_acc"], label="train acc"); ax[1].plot(hist["va_acc"], label="val acc")
        ax[1].plot(hist["va_f1"], label="val macro-F1")
        ax[1].set_title("Accuracy / F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
    a = ax[-1]
    a.imshow(cm, cmap="Blues")
    a.set_xticks(labels); a.set_xticklabels(classes, rotation=45, ha="right")
    a.set_yticks(labels); a.set_yticklabels(classes)
    for i in labels:
        for j in labels:
            a.text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_title("Confusion matrix (test)"); a.set_xlabel("predicted"); a.set_ylabel("true")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def report(y_true, y_pred, classes, title):
    labels = list(range(len(classes)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(f"\n=== {title} ===")
    print(rep)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm_df)
    return rep, cm, cm_df


def main():
    print(f"Loading {CSV_PATH}")
    df = load(CSV_PATH)
    print(f"{len(df):,} rows, {df[SESSION_COL].nunique()} sessions, "
          f"classes: {sorted(df[LABEL_COL].unique())}, channels: {FEATURES}")
    info = describe_sessions(df)

    win = int(WINDOW_SEC * FS)
    step = max(1, int(win * (1 - OVERLAP)))
    X, y_str, groups = make_windows(df, win, step, int(TRIM_SEC * FS))
    classes = sorted(set(y_str))
    y = np.array([classes.index(c) for c in y_str])
    print(f"\n{len(X):,} windows of shape {X.shape[1:]} (window={win}, step={step})")

    if CROSS_VALIDATE:
        splits = cv_splits(info)
        (OUT_DIR / "folds.json").write_text(json.dumps(splits, indent=2))
        all_true, all_pred, accs, f1s = [], [], [], []
        for k, sp in enumerate(splits, 1):
            _, _, best_ep, yte, pred, _, _ = run_split(X, y, groups, sp, len(classes), verbose=False)
            acc = float((yte == pred).mean())
            f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
            accs.append(acc); f1s.append(f1)
            all_true.append(yte); all_pred.append(pred)
            print(f"fold {k}/{N_FOLDS}: {len(sp['test'])} test sessions, best epoch {best_ep}, "
                  f"accuracy {acc:.3f}, macro-F1 {f1:.3f}")
        print(f"\nAccuracy  {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        print(f"Macro-F1  {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        rep, cm, cm_df = report(np.concatenate(all_true), np.concatenate(all_pred), classes,
                                f"POOLED over all {N_FOLDS} folds (every session tested once)")
        (OUT_DIR / "cv_report.txt").write_text(
            f"Accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f}\n"
            f"Macro-F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}\n\n{rep}\n{cm_df.to_string()}")
        plot_results(None, cm, classes, OUT_DIR / "cv_confusion.png")
        print(f"\nSaved folds.json, cv_report.txt and cv_confusion.png to {OUT_DIR}")
        return

    split = split_sessions(info)
    for k, v in split.items():
        print(f"{k}: {len(v)} sessions")
    (OUT_DIR / "split.json").write_text(json.dumps(split, indent=2))

    model, hist, best_ep, yte, pred, mean, std = run_split(X, y, groups, split, len(classes))
    print(f"\nBest epoch (by val macro-F1): {best_ep}")
    torch.save(model.state_dict(), OUT_DIR / "cnn_model.pt")
    np.savez(OUT_DIR / "norm_stats.npz", mean=mean, std=std, classes=np.array(classes))
    rep, cm, cm_df = report(yte, pred, classes, "TEST SET (unseen sessions)")
    (OUT_DIR / "test_report.txt").write_text(rep + "\n" + cm_df.to_string())
    plot_results(hist, cm, classes, OUT_DIR / "results.png")
    print(f"\nSaved model, split.json, report and results.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
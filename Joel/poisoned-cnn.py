
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
CLEAN_CSV = Path(args[0]) if len(args) > 0 else Path(r"C:\Users\Joel_\OneDrive\Documents\GitHub\cleaned_dataset.csv")
POISONED_CSV = Path(args[1]) if len(args) > 1 else Path(r"C:\Users\Joel_\OneDrive\Documents\GitHub\Poisoned-Dataset.csv")  # <-- CHANGE to your poisoned file
FOLDS_JSON = HERE / "cnn_output_v2" / "folds.json"

LABEL_COL = "recording"
SESSION_COL = "activity"
RAW_FEATURES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
FEATURES = RAW_FEATURES + ["acc_mag", "gyro_mag"]

FS = 100
WINDOW_SEC = 2.0
OVERLAP = 0.5
TRIM_SEC = 1.0
FILTER_KERNEL = 5   

N_FOLDS = 5
EPOCHS = 80
BATCH = 64
LR = 5e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 15
if "--quick" in sys.argv:       
    EPOCHS, PATIENCE = 30, 6
SEED = 42
OUT_DIR = HERE / "cnn_output_poison"


rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load(path):
    df = pd.read_csv(path)
    needed = RAW_FEATURES + [LABEL_COL, SESSION_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"{path.name}: missing columns {missing}\nFound: {list(df.columns)}")
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[SESSION_COL] = df[SESSION_COL].astype(str).str.strip()
    return df.reset_index(drop=True)


def add_magnitude(df):
    df["acc_mag"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    df["gyro_mag"] = np.sqrt(df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)
    return df


def median_filter(df):
    out = df.copy()
    out[RAW_FEATURES] = (df.groupby(SESSION_COL, sort=False)[RAW_FEATURES]
                           .transform(lambda s: s.rolling(FILTER_KERNEL, center=True, min_periods=1).median()))
    return add_magnitude(out)


def describe_sessions(df):
    info = (df.groupby(SESSION_COL, sort=False)
              .agg(label=(LABEL_COL, "nunique"), cls=(LABEL_COL, "first"), rows=(RAW_FEATURES[0], "size")))
    if (info["label"] > 1).any():
        sys.exit()
    return info


def make_windows(df, cls_index, win, step, trim):
    
    n = len(cls_index)
    X, y_major, y_centre, g = [], [], [], []
    for sess, d in df.groupby(SESSION_COL, sort=False):
        arr = d[FEATURES].to_numpy(np.float32)
        codes = d[LABEL_COL].map(cls_index).to_numpy().astype(int)
        if len(arr) > 2 * trim + win:
            arr, codes = arr[trim:len(arr) - trim], codes[trim:len(codes) - trim]
        if len(arr) < win:
            continue
        for s in range(0, len(arr) - win + 1, step):
            c = codes[s:s + win]
            X.append(arr[s:s + win])
            y_major.append(np.bincount(c, minlength=n).argmax())
            y_centre.append(c[win // 2])
            g.append(sess)
    return np.stack(X), np.array(y_major), np.array(y_centre), np.array(g)


def carve_val(rest_info):
    val = []
    for cls, grp in rest_info.groupby("cls"):
        s = list(grp.index)
        rng.shuffle(s)
        val += s[:max(1, round(0.15 * len(s))) if len(s) >= 3 else 0]
    val_set = set(val)
    return [s for s in rest_info.index if s not in val_set], val


def get_folds(info):
    if FOLDS_JSON.exists():
        print(f"Using existing folds from {FOLDS_JSON}")
        return json.loads(FOLDS_JSON.read_text())
    print("No folds.json found - creating new folds")
    sess, labels = np.array(info.index), info["cls"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = []
    for tr_idx, te_idx in sgkf.split(sess, labels, groups=sess):
        train, val = carve_val(info.loc[sess[tr_idx]])
        out.append({"train": train, "val": val, "test": list(sess[te_idx])})
    return out


class WinDS(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = x * (1 + 0.15 * torch.randn(x.shape[0], 1))
            x = x + 0.1 * torch.randn_like(x)
            shift = int(torch.randint(-x.shape[1] // 10, x.shape[1] // 10 + 1, (1,)))
            x = torch.roll(x, shift, dims=1)
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


def fit(Xtr, ytr, Xva, yva, n_classes):
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=device)
    tr_loader = DataLoader(WinDS(Xtr, ytr, augment=True), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(WinDS(Xva, yva), batch_size=BATCH)
    model = CNN1D(Xtr.shape[1], n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_ep, best_state, bad = -1.0, 0, None, 0
    for ep in range(1, EPOCHS + 1):
        run_epoch(model, tr_loader, loss_fn, opt)
        run_epoch(model, va_loader, loss_fn)
        vaf1 = f1_score(yva, predict(model, Xva), average="macro", zero_division=0)
        print(".", end="", flush=True)          # one dot per epoch so you can see it is alive
        if vaf1 > best + 1e-4:
            best, best_ep, bad = vaf1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    print(f" done ({ep} ep, best ep {best_ep})", flush=True)
    model.load_state_dict(best_state)
    return model


def train_variant(X, y, idx, n_classes, name=""):
    """Train on the train sessions of one data variant."""
    print(f"  training: {name} ", end="", flush=True)
    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xva, yva = X[idx["val"]], y[idx["val"]]
    mean = Xtr.reshape(-1, Xtr.shape[2]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-8
    prep = lambda a: np.ascontiguousarray(((a - mean) / std).transpose(0, 2, 1), dtype=np.float32)
    model = fit(prep(Xtr), ytr, prep(Xva), yva, n_classes)
    return model, prep


# ------------------------------ MAIN -----------------------------------
CONDITIONS = [
    "1. train -> test (baseline)",
    "2. train -> poisoned test input",
    "3. Clean train -> poisoned test + median filter",
    "4. Poisoned train -> clean test",
    "5. Poisoned train + defences -> clean test",
]


def main():
    for p in (CLEAN_CSV, POISONED_CSV):
        if not p.exists():
            sys.exit(f"File not found: {p}\n")
    clean, pois = load(CLEAN_CSV), load(POISONED_CSV)
    if len(clean) != len(pois) or not (clean[SESSION_COL].to_numpy() == pois[SESSION_COL].to_numpy()).all():
        sys.exit("The two files are different length or session order.")
    info = describe_sessions(clean)
    classes = sorted(clean[LABEL_COL].unique())
    cls_index = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    print(f"{len(clean):,} rows, {len(info)} sessions, classes: {classes}")

    clean_f, pois_f = median_filter(clean), median_filter(pois)
    add_magnitude(clean)
    add_magnitude(pois)

    win = int(WINDOW_SEC * FS)
    step = max(1, int(win * (1 - OVERLAP)))
    trim = int(TRIM_SEC * FS)
    Xc, y_true, _, g = make_windows(clean, cls_index, win, step, trim)       
    Xcf, _, _, _ = make_windows(clean_f, cls_index, win, step, trim)        
    Xp, y_pois_major, y_pois_centre, _ = make_windows(pois, cls_index, win, step, trim)   
    Xpf, _, _, _ = make_windows(pois_f, cls_index, win, step, trim)          
    print(f"{len(Xc):,} windows: {Xc.shape[1:]}")
    print(f"Window label noise if labelled by MIDDLE row:  {(y_pois_centre != y_true).mean():.1%}")
    print(f"Window label noise if labelled by MAJORITY:    {(y_pois_major != y_true).mean():.1%}")

    folds = get_folds(info)
    results = {c: [] for c in CONDITIONS}

    def score(cond, model, prep, Xtest, yte):
        pred = predict(model, prep(Xtest))
        results[cond].append((float((pred == yte).mean()),
                              float(f1_score(yte, pred, average="avg", zero_division=0))))

    for k, sp in enumerate(folds, 1):
        idx = {key: np.isin(g, v) for key, v in sp.items()}
        te = idx["test"]
        yte = y_true[te]

        print(f"fold {k}/{len(folds)}", flush=True)
        m1, p1 = train_variant(Xc, y_true, idx, n_classes, "clean data ")
        score(CONDITIONS[0], m1, p1, Xc[te], yte)
        score(CONDITIONS[1], m1, p1, Xp[te], yte)
        score(CONDITIONS[2], m1, p1, Xpf[te], yte)
        m4, p4 = train_variant(Xp, y_pois_centre, idx, n_classes, "poisoned data 1")
        score(CONDITIONS[3], m4, p4, Xc[te], yte)
        m5, p5 = train_variant(Xpf, y_pois_major, idx, n_classes, "poisoned data + defences")
        score(CONDITIONS[4], m5, p5, Xcf[te], yte)

        line = " | ".join(f"{i + 1}: {results[c][-1][1]:.2f}" for i, c in enumerate(CONDITIONS))
        print(f"fold {k}/{len(folds)} macro-F1 -> {line}", flush=True)

    rows = []
    for c in CONDITIONS:
        a, f = np.array(results[c]).T
        rows.append((c, a.mean(), a.std(), f.mean(), f.std()))
    table = pd.DataFrame(rows, columns=["condition", "acc_mean", "acc_std", "f1_mean", "f1_std"])
    print("\n=== RESULTS (mean +/- std over folds) ===")
    for _, r in table.iterrows():
        print(f"{r['condition']:<52} acc {r['acc_mean']:.3f} +/- {r['acc_std']:.3f}   "
              f"macro-F1 {r['f1_mean']:.3f} +/- {r['f1_std']:.3f}")
    base = table.loc[0, "f1_mean"]
    print(f"\nChange in macro-F1 vs baseline: " +
          ", ".join(f"cond {i + 1}: {table.loc[i, 'f1_mean'] - base:+.3f}" for i in range(1, len(CONDITIONS))))
    table.to_csv(OUT_DIR / "poison_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{i + 1}" for i in range(len(CONDITIONS))]
    ax.bar(labels, table["f1_mean"], yerr=table["f1_std"], capsize=4, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("avg accuracy")
    ax.set_xlabel("condition")
    ax.set_title("Effect of data poisoning on the CNN")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "poison_results.png", dpi=150)


if __name__ == "__main__":
    main()
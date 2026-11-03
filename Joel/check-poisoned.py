import numpy as np, pandas as pd

clean = pd.read_csv(r"C:\Users\Joel_\OneDrive\Documents\GitHub\cleaned_dataset.csv")
pois = pd.read_csv(r"C:\Users\Joel_\OneDrive\Documents\GitHub\Poisoned-Dataset.csv")   # change to the real path
feat = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

lab = clean["recording"] != pois["recording"]
val = (clean[feat] != pois[feat]).any(axis=1)
changed = lab | val
print("label changed:", lab.sum(), "| values changed:", val.sum(),
      "| both:", (lab & val).sum(), "| either:", changed.sum(), flush=True)

print("\nLabel changes (rows = clean label, columns = poisoned label):")
print(pd.crosstab(clean.loc[lab, "recording"], pois.loc[lab, "recording"]), flush=True)

blk = changed.ne(changed.shift()).cumsum()
runs = changed.groupby(blk).agg(["first", "size"])
runs = runs.loc[runs["first"], "size"]

per = changed.groupby(clean["activity"]).mean()
print(per.describe().round(3), flush=True)

diff = (pois.loc[val, feat] - clean.loc[val, feat]).abs().mean().round(3)
print("\nMean absolute change on altered rows:", diff.to_dict())

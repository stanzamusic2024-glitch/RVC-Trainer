# -*- coding: utf-8 -*-
"""Helper script: trains the FAISS index for an RVC experiment.

Usage:
    python train_index_helper.py <exp_dir_name> <version> <rvc_home>
"""
import os, sys, traceback

rvc_home = sys.argv[3] if len(sys.argv) > 3 else os.getcwd()
os.chdir(rvc_home)
sys.path.insert(0, rvc_home)

exp_dir_name = sys.argv[1]
version = sys.argv[2]
exp_dir = os.path.join(rvc_home, "logs", exp_dir_name)

import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans

feature_dir = os.path.join(
    exp_dir, "3_feature256" if version == "v1" else "3_feature768"
)

def log(msg):
    print(msg, flush=True)

if not os.path.exists(feature_dir):
    log("Feature-Verzeichnis nicht gefunden: " + feature_dir)
    sys.exit(1)

listdir_res = list(os.listdir(feature_dir))
if len(listdir_res) == 0:
    log("Keine Feature-Dateien gefunden.")
    sys.exit(1)

infos = []
npys = []
for name in sorted(listdir_res):
    phone = np.load(os.path.join(feature_dir, name))
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
big_npy = big_npy[big_npy_idx]

if big_npy.shape[0] > 2e5:
    infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
    print("\n".join(infos), flush=True)
    try:
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * 4,
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )
    except Exception:
        info = traceback.format_exc()
        print(info, flush=True)

np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
infos.append("%s,%s" % (big_npy.shape, n_ivf))
print("\n".join(infos), flush=True)

index = faiss.index_factory(256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf)
infos.append("training")
print("\n".join(infos), flush=True)
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
index.train(big_npy)
faiss.write_index(
    index,
    os.path.join(exp_dir, "trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
                 % (n_ivf, index_ivf.nprobe, exp_dir_name, version)),
)

infos.append("adding")
print("\n".join(infos), flush=True)
batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i : i + batch_size_add])
faiss.write_index(
    index,
    os.path.join(exp_dir, "added_IVF%s_Flat_nprobe_%s_%s_%s.index"
                 % (n_ivf, index_ivf.nprobe, exp_dir_name, version)),
)
infos.append("Successful Index Construction: added_IVF%s_Flat_nprobe_%s_%s_%s.index"
             % (n_ivf, index_ivf.nprobe, exp_dir_name, version))
print("\n".join(infos), flush=True)
print("INDEX_DONE", flush=True)

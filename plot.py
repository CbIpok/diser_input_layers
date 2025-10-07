# Подробный вывод по КАЖДОМУ файлу RMSE (без усреднения)
# ЧИТАЕТ только .npy из output/rmse_mean и печатает таблицу.

import os, re, glob, numpy as np

files = sorted(glob.glob(r"output/rmse_mean/*.npy"))
rx = re.compile(r"rmse_mean__i_(?P<i>\d+)__func_(?P<func>.+)__recon_sigma_(?P<sigma>[^.]+)\.npy$")

rows = []
for f in files:
    name = os.path.basename(f)
    m = rx.match(name)
    if not m:
        continue
    i = int(m.group("i"))
    func = m.group("func")
    sigma = m.group("sigma")
    a = np.load(f)
    rows.append({
        "i": i,
        "func": func,
        "sigma": sigma,
        "mean": float(np.nanmean(a)),
        "median": float(np.nanmedian(a)),
        "p95": float(np.nanpercentile(a,25)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "file": name
    })

# сортировка: по i, затем по func, затем по sigma
rows.sort(key=lambda r: (r["i"], r["func"], r["sigma"]))

# печать таблицы
hdr = f"{'i':>3} | {'func':<18} | {'sigma':<6} | {'mean':>10} | {'median':>10} | {'p95':>10} | {'min':>10} | {'max':>10} | file"
print(hdr)
print("-"*len(hdr))
for r in rows:
    print(f"{r['i']:>3} | {r['func']:<18} | {r['sigma']:<6} | "
          f"{r['mean']:>10.6f} | {r['median']:>10.6f} | {r['p95']:>10.6f} | "
          f"{r['min']:>10.6f} | {r['max']:>10.6f} | {r['file']}")

"""Iz site.npz (LiDAR) naredi site_scene.npz, ki ga uporablja turtle_dataset.py.

  python make_site.py        (potrebuje samo numpy)

Lokalne koordinate: izhodisce (0, 0) = tocka 45.94005 N, 14.55201 E (D96/TM E=465263.6, N=89011.0),
x proti vzhodu, y proti severu, metri. Visina 0 = teren na sredini travnika.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, 'site.npz'))
dmr, chm, cnt = z['dmr'].astype(np.float64), z['chm'].astype(np.float64), z['cnt']
ny, nx = dmr.shape
HALF = nx / 2.0                     # 120 m
res = float(z['res'])


def box(a, r):
    """povprecje v oknu (2r+1)^2"""
    p = np.pad(a, r, mode='edge')
    out = np.zeros_like(a, dtype=np.float64)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / (2 * r + 1) ** 2


def components(mask):
    """oznake povezanih obmocij (4-sosedstvo), brez scipy"""
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    H, W = mask.shape
    for sy in range(H):
        for sx in range(W):
            if mask[sy, sx] and not lab[sy, sx]:
                n += 1
                st = [(sy, sx)]; lab[sy, sx] = n
                while st:
                    y, x = st.pop()
                    for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                            lab[yy, xx] = n; st.append((yy, xx))
    return lab, n


def keep_big(mask, min_area):
    lab, n = components(mask)
    if n == 0:
        return mask
    areas = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, bool); keep[1:] = areas[1:] >= min_area
    return keep[lab]


def dist_inside(mask, maxd=8):
    """priblizna razdalja (m) od roba navznoter (zaporedna erozija)"""
    d = np.zeros(mask.shape, np.float64)
    m = mask.copy()
    for k in range(1, maxd + 1):
        m = m & (box(m.astype(float), 1) > 0.999)
        d += m
    return d


# ---- mlaka: celice brez odboja (voda) + lokvanji vmes -> zgladi, obdrzi velika obmocja
water0 = cnt == 0
pond = box(water0.astype(float), 3) > 0.25
pond = keep_big(pond, 400)
pond = box(pond.astype(float), 1) > 0.5
WL = float(np.median(dmr[pond])) if pond.any() else float(dmr.min())
# lokvanji vrnejo odboj -> dodaj odprte celice na visini gladine, povezane z vodo
flat = (dmr <= WL + 0.12) & (chm < 1.0) & (np.arange(nx)[None, :] - HALF < -45)   # samo zahodno od roba vode
lab, n = components(flat | pond)
ids = np.unique(lab[pond]); ids = ids[ids > 0]
pond = np.isin(lab, ids)
pond = box(pond.astype(float), 1) > 0.5
print('mlaka celic:', int(pond.sum()), ' gladina (abs):', round(WL, 2))

# ---- travnik: brez rastlinja, ni voda; zgladi in obdrzi glavna obmocja
open_ = (chm < 0.7) & ~pond
open_ = box(open_.astype(float), 1) > 0.6
meadow = keep_big(open_, 600)
meadow_soft = np.clip(box(meadow.astype(float), 2) * 1.2, 0, 1)
print('travnik celic:', int(meadow.sum()))

# ---- referencna visina: sredina travnika
my, mx = np.nonzero(meadow)
zref = float(np.median(dmr[meadow]))

# ---- teren z dnom mlake
terr = dmr.copy()
din = dist_inside(pond, 10)
depth = 1.4 * np.clip(din / 6.0, 0, 1) ** 0.8
terr = np.where(pond, np.minimum(terr, WL - 0.08 - depth), terr)
terr = box(terr, 1) * 0.5 + terr * 0.5          # mehkejsi prehodi brez izgube oblike
terr = np.where(pond, np.minimum(terr, WL - 0.08 - depth), terr)

# ---- drevesa: vrhovi krosenj (lokalni maksimumi CHM)
chs = box(chm, 1)
peak = np.ones_like(chs, bool)
R = 2
p = np.pad(chs, R, mode='constant', constant_values=-1)
for dy in range(2 * R + 1):
    for dx in range(2 * R + 1):
        if dy == R and dx == R:
            continue
        peak &= chs >= p[dy:dy + ny, dx:dx + nx]
trees = []
ty, tx = np.nonzero(peak & (chs > 3.0) & ~pond)
order = np.argsort(-chs[ty, tx])
taken = np.zeros_like(chs, bool)
for k in order:
    y, x = ty[k], tx[k]
    if taken[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].any():
        continue
    taken[y, x] = True
    h = float(chm[y, x])
    r = float(np.clip(0.16 * h + 1.0, 1.5, 6.0))
    trees.append((x + 0.5 - HALF, y + 0.5 - HALF, h, r))
# zapolni gozd: visoko rastlinje, ki ga se ne pokriva nobena krosnja, dobi dodatno drevo
cov = np.zeros_like(chs, bool)
YY, XX = np.mgrid[0:ny, 0:nx]
def cover(x, y, r):
    i0, i1 = int(max(y - r - 1, 0)), int(min(y + r + 2, ny)); j0, j1 = int(max(x - r - 1, 0)), int(min(x + r + 2, nx))
    cov[i0:i1, j0:j1] |= (XX[i0:i1, j0:j1] - x) ** 2 + (YY[i0:i1, j0:j1] - y) ** 2 <= (1.0 * r) ** 2
for x, y, h, r in trees:
    cover(x + HALF - 0.5, y + HALF - 0.5, r)
high = (chs > 4.0) & ~pond
for it in range(6):
    gy_, gx_ = np.nonzero(high & ~cov)
    if len(gy_) == 0:
        break
    for k in np.random.default_rng(it).permutation(len(gy_)):
        y, x = gy_[k], gx_[k]
        if cov[y, x]:
            continue
        h = float(chm[y, x]); r = float(np.clip(0.15 * h + 1.5, 2.5, 5.0))
        trees.append((x + 0.5 - HALF, y + 0.5 - HALF, h, r)); cover(x, y, r)
trees = np.array(trees, np.float32)
print('dreves:', len(trees))

# ---- grmovje: nizko rastlinje 0.7-3 m (redcenje)
rng = np.random.default_rng(1)
sy, sx = np.nonzero((chm > 0.7) & (chm < 3.0) & ~pond)
pick = rng.random(len(sy)) < 0.25
shrubs = np.array([(x + rng.uniform(0, 1) - HALF, y + rng.uniform(0, 1) - HALF, float(chm[y, x]))
                   for y, x in zip(sy[pick], sx[pick])], np.float32)
print('grmov:', len(shrubs))

# ---- kolovoz (trasa s satelitskega posnetka; okno: X od zahoda, Y od severa v metrih)
track_win = [(70, 62), (80, 68), (92, 80), (118, 96), (143, 114), (156, 130), (159, 143),
             (156, 157), (150, 169), (141, 182), (134, 200), (130, 220), (128, 240)]
track = np.array([(X - HALF, HALF - Yt) for X, Yt in track_win], np.float32)

# ---- obmocje rdecega kroga (elipsa)
aoi = np.array([(125.6 - HALF, HALF - 115.4), 80.0, 53.0], dtype=object)

np.savez_compressed(os.path.join(HERE, 'site_scene.npz'),
                    terrain=(terr - zref).astype(np.float32), pond=pond.astype(np.uint8),
                    meadow=meadow_soft.astype(np.float32), chm=chm.astype(np.float16),
                    trees=trees, shrubs=shrubs, track=track,
                    aoi_center=np.array([125.6 - HALF, HALF - 115.4], np.float32), aoi_radii=np.array([80.0, 53.0], np.float32),
                    water_level=np.float32(WL - zref), res=np.float32(res), half=np.float32(HALF),
                    origin_d96=np.array([z['ce'], z['cn']]), zref=np.float32(zref))
print('zref', round(zref, 2), ' gladina lokalno', round(WL - zref, 2), ' teren lokalno', round(float((terr - zref).min()), 2), round(float((terr - zref).max()), 2))

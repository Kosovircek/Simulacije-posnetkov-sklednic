"""Obdelava ARSO LiDAR podatkov za lokacijo bajerja (navaden Python z laspy[lazrs] + numpy).

  python process_lidar.py

Izhod (v tej mapi):
  site.npz        dmr (teren), dsm (najvisja tocka), chm (visina rastlinja), water (delez vode), meta
  site_*.png      pregledne karte
"""
import os, sys, struct, zlib
import numpy as np
import laspy

HERE = os.path.dirname(os.path.abspath(__file__))
CE, CN = 465263.6, 89011.0      # tocka (45.94005 N, 14.55201 E) v D96/TM
HALF = 120                      # okno 240 x 240 m
RES = 1.0


def write_png(path, img):
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    h, w, _ = img.shape
    raw = np.hstack([np.zeros((h, 1), np.uint8), img.reshape(h, w * 3)]).tobytes()
    ch = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + ch(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                           ch(b'IDAT', zlib.compress(raw)) + ch(b'IEND', b''))


x0, x1 = CE - HALF, CE + HALF
y0, y1 = CN - HALF, CN + HALF
nx, ny = int(2 * HALF / RES), int(2 * HALF / RES)

# --- DMR (teren), 1 m mreza x;y;z
d = np.vstack([np.loadtxt(os.path.join(HERE, f), delimiter=';') for f in ('TM1_465_89.asc', 'TM1_465_88.asc')])
sel = (d[:, 0] >= x0) & (d[:, 0] < x1) & (d[:, 1] >= y0) & (d[:, 1] < y1)
d = d[sel]
dmr = np.full((ny, nx), np.nan, np.float32)
ix = ((d[:, 0] - x0) / RES).astype(int); iy = ((d[:, 1] - y0) / RES).astype(int)
ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
dmr[iy[ok], ix[ok]] = d[ok, 2]
print('DMR min/max', np.nanmin(dmr), np.nanmax(dmr), 'nan', int(np.isnan(dmr).sum()))

# --- oblak tock: DSM (najvisja tocka) in voda
X, Y, Z, cls = [], [], [], []
for f in ('TM_465_89.laz', 'TM_465_88.laz'):
    las = laspy.read(os.path.join(HERE, f))
    X.append(np.asarray(las.x)); Y.append(np.asarray(las.y)); Z.append(np.asarray(las.z))
    cls.append(np.asarray(las.classification))
X, Y, Z, cls = (np.concatenate(v) for v in (X, Y, Z, cls))
print('razredi tock:', {int(c): int((cls == c).sum()) for c in np.unique(cls)})
sel = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1) & (cls != 7)
X, Y, Z, cls = X[sel], Y[sel], Z[sel], cls[sel]
ix = ((X - x0) / RES).astype(int); iy = ((Y - y0) / RES).astype(int)
lin = iy * nx + ix
dsm = np.full(nx * ny, -1e9, np.float32)
np.maximum.at(dsm, lin, Z.astype(np.float32))
dsm[dsm < -1e8] = np.nan
dsm = dsm.reshape(ny, nx)
cnt = np.bincount(lin, minlength=nx * ny).reshape(ny, nx)
wat = np.bincount(lin, weights=(cls == 9).astype(float), minlength=nx * ny).reshape(ny, nx)
water = np.where(cnt > 0, wat / np.maximum(cnt, 1), 0).astype(np.float32)
chm = np.clip(np.nan_to_num(dsm - dmr, nan=0.0), 0, 60).astype(np.float32)
print('CHM max', chm.max(), 'voda (delez celic >0.5):', float((water > 0.5).mean()), 'tock brez povratka (celice cnt=0):', int((cnt == 0).sum()))

np.savez_compressed(os.path.join(HERE, 'site.npz'), dmr=dmr, dsm=dsm, chm=chm, water=water, cnt=cnt,
                    x0=x0, y0=y0, res=RES, ce=CE, cn=CN)

# --- pregledne karte (sever zgoraj)
def flip(a):
    return a[::-1]
gy, gx = np.gradient(np.nan_to_num(dmr, nan=np.nanmean(dmr)))
shade = np.clip(0.5 + (-gx * 0.7 + gy * 0.7) * 1.5, 0, 1)
z = (dmr - np.nanmin(dmr)) / (np.nanmax(dmr) - np.nanmin(dmr) + 1e-6)
rel = np.stack([z * 0.6 + shade * 0.4] * 3, -1)
write_png(os.path.join(HERE, 'site_relief.png'), flip((np.nan_to_num(rel) * 255).astype(np.uint8)))
c = np.clip(chm / 30.0, 0, 1)
img = np.stack([c * 0.2, 0.3 + c * 0.7, c * 0.2], -1)
img[water > 0.3] = (0.1, 0.3, 0.9)
img[chm < 0.5] = (0.85, 0.85, 0.5)
img[(water > 0.3)] = (0.1, 0.3, 0.9)
write_png(os.path.join(HERE, 'site_vegetation.png'), flip((img * 255).astype(np.uint8)))
print('ok')

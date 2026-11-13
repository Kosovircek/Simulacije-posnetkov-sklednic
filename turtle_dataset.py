"""
Sinteticni dataset: zelva na travniku (RGB + termalna slika + maska + oznake)
==========================================================================

Blender 4.2+ / 5.0.  Uporaba (iz ukazne vrstice):

  1) Zgradi sceno in shrani .blend:
     blender -b -P turtle_dataset.py -- --build --blend turtle_meadow.blend

  2) Generiraj slike (npr. 1000):
     blender -b turtle_meadow.blend -P turtle_dataset.py -- --render 1000 --out dataset

     Dodatno:  --seed 7   --start 1000 (nadaljuj stevilcenje)   --cycles

  3) Animacija (ena scena, zelva gre od mlake cez travnik skoraj do kamere) -> mp4:
     blender -b turtle_meadow.blend -P turtle_dataset.py -- --animate --seed 3 --out animations

     Dodatno:  --fps 10   --stop-dist 5 (m od kamere, kjer se ustavi)   --keep-frames (obdrzi slicice + YOLO oznake)
     Hitro: ozadje se zrenderira enkrat, nato vsako slicico samo izsek okoli zelve in sence.

     Predogled poti (cela pot se simulira, zrenderira se samo 10 slicic, brez videa):
     blender -b turtle_meadow.blend -P turtle_dataset.py -- --preview 10 --seed 3 --out animations
     Izhod: animations/anim_0003/ rgb.mp4, thermal.mp4, combined.mp4, track.csv (pot zelve po slicicah),
            info.json, scene.blend (animirana scena za ogled v Blenderju)

Izhod (v mapi --out).  Ime slike = koordinate zelve: x+12.345_y-3.210_00012.png
(X, Y v metrih v koordinatnem sistemu travnika, zadnja stevilka je zaporedna st. slike)
  rgb/<ime>.png        normalna slika
  thermal/<ime>.png    termalna slika (8-bit sivinska, white-hot)
  mask/<ime>.png       segmentacijska maska vidnega dela zelve
  combined/<ime>.png   RGB (levo) + termalna (desno) v eni sliki - za pregled
  labels/<ime>.txt     YOLO oznaka (class cx cy w h, normalizirano - velja za RGB in termalno)
  meta/<ime>.json      vse: koordinate zelve v svetu, kamera, bbox, parametri, temperature
  annotations.csv          povzetek vseh slik

Koordinatni sistem: travnik je 100 m (X) x 50 m (Y) = 5000 m2 = 0.5 ha, sredisce (0,0).
Mlaka je ob vzhodnem robu (+X).  Enote: metri.
"""

import bpy, bmesh, math, random, os, sys, json, csv, struct, zlib, colorsys, time
import numpy as np
from mathutils import Vector, Matrix, Euler, Quaternion, noise
from bpy_extras.object_utils import world_to_camera_view

# ---------------------------------------------------------------- CONFIG
MEADOW_X, MEADOW_Y = 100.0, 50.0          # 0.5 ha
GRID = 0.25                               # resolucija terena (m)
WATER_LEVEL = -0.35
POND = dict(cx=38.0, cy=6.0, rx=9.0, ry=12.0, depth=1.1)

RGB_RES = (1280, 1024)                    # enako razmerje stranic kot termalna!
THERMAL_RES = (640, 512)
# FIKSNA kamera (enaka na vseh slikah) - na sredini travnika, 5 m visoko, gleda posevno cez travnik
CAM_LOCATION = (0.0, 0.0, 5.0)
CAM_TILT_DEG = 60.0                       # odklon od navpicnice (0 = naravnost dol, 90 = vodoravno)
CAM_YAW_DEG = -90.0                       # -90 = gleda proti +X (proti mlaki), 0 = proti +Y
CAM_FOCAL = 24.0                          # mm (senzor 36 mm)
MAX_TURTLE_DIST = 25.0                    # najvecja oddaljenost zelve od kamere (m), da ni premajhna

TUFT_DENSITY = (0.0, 45.0)                # redki 3D sopi trave / m2 (vecina trave je tekstura)
TUFT_NEAR, TUFT_FAR = 6.0, 25.0           # sopi so blizu kamere, dlje od TUFT_FAR jih ni
MAX_GRASS_INSTANCES = 150_000             # varovalka za VRAM
TURTLE_LEN = (0.15, 0.25)                 # dolzina oklepa (m), cca 20 cm
NEAR_POND_PROB = 0.25                     # verjetnost, da je zelva ob mlaki
MIN_VISIBLE_PIXELS = 25                   # ce je zelva bolj skrita v travi -> nov poskus
NEGATIVE_PROB = 0.0                       # delez slik BREZ zelve (npr. 0.1 za negativne primere)
SAMPLES_RGB, SAMPLES_THERMAL, SAMPLES_MASK = 32, 16, 4

# ---------------------------------------------------------------- utils
def U(r, a, b): return r.uniform(a, b)
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def smooth(t): t = clamp(t); return t * t * (3 - 2 * t)
def hsv(h, s, v): return colorsys.hsv_to_rgb(h % 1.0, clamp(s), max(v, 0.0))
def scl(c, k): return tuple(clamp(x * k, 0, 1) for x in c)


def _sock(sockets, key):
    if isinstance(key, int):
        return sockets[key]
    ident = [s for s in sockets if s.identifier == key]
    named = [s for s in sockets if s.name == key]
    for group in (ident, named):
        en = [s for s in group if getattr(s, "enabled", True)]
        if en:
            return en[0]
    if ident: return ident[0]
    if named: return named[0]
    raise KeyError(key)


class NT:
    """Majhen pomocnik za gradnjo node dreves."""
    def __init__(self, tree):
        self.t, self.N, self.L = tree, tree.nodes, tree.links

    def node(self, typ, inputs=None, name=None, **props):
        n = self.N.new(typ)
        for k, v in props.items():
            setattr(n, k, v)
        if name:
            n.name = n.label = name
        for k, v in (inputs or {}).items():
            self.set(_sock(n.inputs, k), v)
        return n

    def set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.L.new(v, sock)
        else:
            if sock.type == 'RGBA' and isinstance(v, (tuple, list)) and len(v) == 3:
                v = (*v, 1.0)
            elif sock.type == 'RGBA' and isinstance(v, (int, float)):
                v = (v, v, v, 1.0)
            sock.default_value = v

    def out(self, n, key=0): return _sock(n.outputs, key)

    def value(self, name, v):
        n = self.N.get(name)
        if n is None:
            n = self.node('ShaderNodeValue', name=name)
            n.outputs[0].default_value = v
        return n.outputs[0]

    def rgb(self, name, c):
        n = self.N.get(name)
        if n is None:
            n = self.node('ShaderNodeRGB', name=name)
            n.outputs[0].default_value = (*c, 1.0)
        return n.outputs[0]

    def math(self, op, a, b=0.0, c=0.0, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.set(n.inputs[0], a); self.set(n.inputs[1], b); self.set(n.inputs[2], c)
        return n.outputs[0]

    def mul(self, *xs):
        r = xs[0]
        for x in xs[1:]:
            r = self.math('MULTIPLY', r, x)
        return r

    def maprange(self, v, a, b, c, d, clamp=True):
        n = self.node('ShaderNodeMapRange', clamp=clamp)
        for k, x in zip(('Value', 'From Min', 'From Max', 'To Min', 'To Max'), (v, a, b, c, d)):
            self.set(_sock(n.inputs, k), x)
        return _sock(n.outputs, 'Result')

    def mix(self, fac, a, b, dtype='RGBA', blend='MIX'):
        n = self.node('ShaderNodeMix', data_type=dtype)
        n.blend_type = blend
        suf = {'RGBA': 'Color', 'FLOAT': 'Float', 'VECTOR': 'Vector'}[dtype]
        self.set(_sock(n.inputs, 'Factor_Float'), fac)
        self.set(_sock(n.inputs, 'A_' + suf), a)
        self.set(_sock(n.inputs, 'B_' + suf), b)
        return _sock(n.outputs, 'Result_' + suf)

    def combine(self, x, y, z):
        n = self.node('ShaderNodeCombineXYZ')
        self.set(n.inputs[0], x); self.set(n.inputs[1], y); self.set(n.inputs[2], z)
        return n.outputs[0]

    def separate(self, v):
        n = self.node('ShaderNodeSeparateXYZ'); self.set(n.inputs[0], v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def noise(self, vec, scale, detail=2.0, w=None, rough=0.5):
        n = self.node('ShaderNodeTexNoise')
        n.noise_dimensions = '4D' if w is not None else '3D'
        self.set(_sock(n.inputs, 'Vector'), vec)
        self.set(_sock(n.inputs, 'Scale'), scale)
        self.set(_sock(n.inputs, 'Detail'), detail)
        self.set(_sock(n.inputs, 'Roughness'), rough)
        if w is not None:
            self.set(_sock(n.inputs, 'W'), w)
        return _sock(n.outputs, 'Fac')


# ---------------------------------------------------------------- terrain height (deterministic)
def _raw_height(x, y):
    h = 0.18 * noise.noise(Vector((x * 0.03, y * 0.03, 0.5)))
    h += 0.06 * noise.noise(Vector((x * 0.15, y * 0.15, 1.7)))
    h += 0.015 * noise.noise(Vector((x * 0.9, y * 0.9, 3.3)))
    e = min(MEADOW_X / 2 - abs(x), MEADOW_Y / 2 - abs(y))
    h *= smooth(e / 6.0)
    d = ((x - POND['cx']) / POND['rx']) ** 2 + ((y - POND['cy']) / POND['ry']) ** 2
    d *= 1.0 + 0.18 * noise.noise(Vector((x * 0.2, y * 0.2, 9.1)))
    if d < 1.6:
        h -= POND['depth'] * smooth((1.6 - d) / 1.3)
    return h


def terrain_height(x, y):
    """Visina terena v tocki (bilinearno iz mreze - natancno kot mesh)."""
    hx, hy = MEADOW_X / 2, MEADOW_Y / 2
    if abs(x) >= hx or abs(y) >= hy:
        return 0.0
    fx, fy = (x + hx) / GRID, (y + hy) / GRID
    i, j = int(fx), int(fy)
    tx, ty = fx - i, fy - j
    x0, y0 = -hx + i * GRID, -hy + j * GRID
    h00 = _raw_height(x0, y0); h10 = _raw_height(x0 + GRID, y0)
    h01 = _raw_height(x0, y0 + GRID); h11 = _raw_height(x0 + GRID, y0 + GRID)
    return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty


def terrain_normal(x, y, e=0.1):
    dx = terrain_height(x + e, y) - terrain_height(x - e, y)
    dy = terrain_height(x, y + e) - terrain_height(x, y - e)
    return Vector((-dx / (2 * e), -dy / (2 * e), 1.0)).normalized()


# ---------------------------------------------------------------- materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    m.node_tree.nodes.clear()
    return m, NT(m.node_tree)


def th_gray(B, key, scale=1.0, objrand=False, pos=None):
    """Termalna 'temperatura' (v sivinah 0..1) z prostorskim sumom."""
    base = B.value(f'TH_{key}_B', 0.5)
    var = B.value(f'TH_{key}_V', 0.03)
    if pos is None:
        pos = B.out(B.node('ShaderNodeNewGeometry'), 'Position')
    nz = B.noise(pos, scale, detail=3.0)
    g = B.math('MULTIPLY_ADD', B.math('MULTIPLY_ADD', nz, 2.0, -1.0), var, base)
    if objrand:
        r = B.out(B.node('ShaderNodeObjectInfo'), 'Random')
        g = B.math('MULTIPLY_ADD', B.math('MULTIPLY_ADD', r, 2.0, -1.0), B.value(f'TH_{key}_R', 0.02), g)
    return g


def finish(m, B, rgb_shader, thermal_gray, mask):
    """RGB / termalni / maska nacin preklapljamo z vrednostma MODE_T in MODE_M."""
    mt, mm, sol = B.value('MODE_T', 0.0), B.value('MODE_M', 0.0), B.value('SOLAR', 0.05)
    tcol = B.node('ShaderNodeCombineColor')   # R,G = temperatura, B = maska zelve
    B.set(tcol.inputs[0], thermal_gray); B.set(tcol.inputs[1], thermal_gray); tcol.inputs[2].default_value = mask
    em_t = B.node('ShaderNodeEmission'); B.L.new(tcol.outputs[0], em_t.inputs['Color'])
    em_t.inputs['Strength'].default_value = 1.0
    scol = B.node('ShaderNodeCombineColor')   # soncno segrevanje samo v R,G (B ostane cista maska)
    B.set(scol.inputs[0], sol); B.set(scol.inputs[1], sol); scol.inputs[2].default_value = 0.0
    dif = B.node('ShaderNodeBsdfDiffuse'); B.L.new(scol.outputs[0], dif.inputs['Color'])
    add = B.node('ShaderNodeAddShader')
    B.L.new(em_t.outputs[0], add.inputs[0]); B.L.new(dif.outputs[0], add.inputs[1])
    em_m = B.node('ShaderNodeEmission'); em_m.inputs['Color'].default_value = (mask, mask, mask, 1)
    em_m.inputs['Strength'].default_value = 1.0
    m1 = B.node('ShaderNodeMixShader'); B.set(m1.inputs[0], mt)
    B.L.new(rgb_shader, m1.inputs[1]); B.L.new(add.outputs[0], m1.inputs[2])
    m2 = B.node('ShaderNodeMixShader'); B.set(m2.inputs[0], mm)
    B.L.new(m1.outputs[0], m2.inputs[1]); B.L.new(em_m.outputs[0], m2.inputs[2])
    o = B.node('ShaderNodeOutputMaterial'); B.L.new(m2.outputs[0], o.inputs['Surface'])
    return m


def principled(B, color, rough=0.5, normal=None, translucent=0.0):
    p = B.node('ShaderNodeBsdfPrincipled')
    B.set(p.inputs['Base Color'], color); B.set(p.inputs['Roughness'], rough)
    if normal is not None:
        B.set(p.inputs['Normal'], normal)
    if translucent <= 0:
        return p.outputs[0]
    t = B.node('ShaderNodeBsdfTranslucent'); B.set(t.inputs['Color'], color)
    ms = B.node('ShaderNodeMixShader'); ms.inputs[0].default_value = translucent
    B.L.new(p.outputs[0], ms.inputs[1]); B.L.new(t.outputs[0], ms.inputs[2])
    return ms.outputs[0]


def mat_grass():
    m, B = new_mat('TD_Grass')
    pos = B.out(B.node('ShaderNodeNewGeometry'), 'Position')
    z = B.separate(B.out(B.node('ShaderNodeTexCoord'), 'Generated'))[2]
    rnd = B.out(B.node('ShaderNodeObjectInfo'), 'Random')
    rnd2 = B.math('FRACT', B.math('MULTIPLY', rnd, 17.31))
    f1 = B.maprange(B.noise(pos, B.value('G_NSCALE', 0.4), 3.0), 0.3, 0.7, 0.0, 1.0)
    col = B.mix(f1, B.rgb('G_COL_A', (0.05, 0.12, 0.02)), B.rgb('G_COL_B', (0.08, 0.14, 0.03)))
    hs = B.node('ShaderNodeHueSaturation')
    B.set(hs.inputs['Hue'], B.math('MULTIPLY_ADD', B.math('SUBTRACT', rnd2, 0.5), B.value('G_HUEJ', 0.04), 0.5))
    B.set(hs.inputs['Value'], B.math('MULTIPLY_ADD', B.math('SUBTRACT', rnd, 0.5), B.value('G_VALJ', 0.5), 1.0))
    B.set(hs.inputs['Color'], col); col = hs.outputs[0]
    n2 = B.noise(pos, B.value('G_DSCALE', 0.2), 2.0)
    thr = B.mul(B.value('G_DRY_FRAC', 0.1), n2, 2.0)
    dry = B.math('LESS_THAN', rnd, thr)
    col = B.mix(dry, col, B.rgb('G_DRY', (0.25, 0.2, 0.08)))
    tipf = B.mul(B.math('POWER', z, 2.5), B.value('G_TIP_AMT', 0.3))
    col = B.mix(tipf, col, B.rgb('G_TIP', (0.2, 0.22, 0.08)))
    dark = B.maprange(z, 0.0, 0.6, B.value('G_BASE_DARK', 0.35), 1.0)
    col = B.mix(1.0, col, dark, blend='MULTIPLY')
    sh = principled(B, col, 0.55, translucent=0.3)
    return finish(m, B, sh, th_gray(B, 'veg', 3.0, objrand=True), 0.0)


def mat_simple(name, colkey, col, th_key, rough=0.5, translucent=0.0, jitter=0.3, nscale=8.0,
               objrand_th=True, bump=0.0):
    m, B = new_mat(name)
    pos = B.out(B.node('ShaderNodeNewGeometry'), 'Position')
    rnd = B.out(B.node('ShaderNodeObjectInfo'), 'Random')
    c = B.rgb(colkey, col)
    nz = B.noise(pos, nscale, 3.0)
    k = B.math('ADD', B.math('MULTIPLY', B.math('SUBTRACT', nz, 0.5), 0.8),
               B.math('MULTIPLY_ADD', B.math('SUBTRACT', rnd, 0.5), jitter * 2, 1.0))
    c = B.mix(1.0, c, k, blend='MULTIPLY')
    normal = None
    if bump > 0:
        bn = B.node('ShaderNodeBump'); bn.inputs['Strength'].default_value = bump
        B.set(bn.inputs['Height'], B.noise(pos, nscale * 3, 4.0)); normal = bn.outputs[0]
    sh = principled(B, c, rough, normal, translucent)
    return finish(m, B, sh, th_gray(B, th_key, 4.0, objrand=objrand_th), 0.0)


def mat_ground():
    m, B = new_mat('TD_Ground')
    pos = B.out(B.node('ShaderNodeNewGeometry'), 'Position')
    pz = B.separate(pos)[2]
    nz = B.noise(pos, B.value('N_SCALE', 0.15), 2.0, w=B.value('N_W', 0.0))
    thr = B.value('BARE_THR', 0.3)
    grassy = B.maprange(nz, B.math('SUBTRACT', thr, 0.04), B.math('ADD', thr, 0.04), 0.0, 1.0)
    fine = B.maprange(B.noise(pos, 6.0, 4.0), 0.3, 0.7, 0.0, 1.0)
    soil = B.mix(fine, B.rgb('S_COL_D', (0.03, 0.02, 0.01)), B.rgb('S_COL', (0.07, 0.045, 0.02)))

    # ---- proceduralna trava (tekstura) ----
    # velike krpe barve
    f1 = B.maprange(B.noise(pos, B.value('G_NSCALE', 0.4), 3.0), 0.3, 0.7, 0.0, 1.0)
    gcol = B.mix(f1, B.rgb('G_COL_A', (0.08, 0.14, 0.03)), B.rgb('G_COL_B', (0.1, 0.16, 0.04)))
    # suhe krpe
    dn = B.noise(pos, B.value('G_DSCALE', 0.2), 3.0)
    dfac = B.math('MULTIPLY', B.maprange(dn, 0.45, 0.75, 0.0, 1.0), B.math('MULTIPLY', B.value('G_DRY_FRAC', 0.1), 1.6), clamp=True)
    gcol = B.mix(dfac, gcol, B.rgb('G_DRY', (0.25, 0.2, 0.08)))
    # 'bilke': podolgovat sum, zasukan v smer polegnjenosti trave
    mp = B.node('ShaderNodeMapping')
    B.set(_sock(mp.inputs, 'Vector'), pos)
    B.set(_sock(mp.inputs, 'Rotation'), B.combine(0.0, 0.0, B.value('LEAN_DIR', 0.0)))
    B.set(_sock(mp.inputs, 'Scale'), B.combine(B.value('GT_STRETCH', 0.15), 1.0, 1.0))
    sv = mp.outputs[0]
    fine_s = B.value('GT_FINE', 120.0)
    st1 = B.noise(sv, fine_s, 2.0, rough=0.6)
    st2 = B.noise(sv, B.math('MULTIPLY', fine_s, 2.3), 1.0)
    con = B.value('GT_CONTRAST', 0.4)
    k1 = B.maprange(st1, 0.25, 0.75, B.math('SUBTRACT', 1.0, con), B.math('ADD', 1.0, con))
    gcol = B.mix(1.0, gcol, k1, blend='MULTIPLY')
    # svetle suhe bilke (slama)
    straw = B.math('MULTIPLY', B.maprange(st2, 0.66, 0.72, 0.0, 1.0), B.value('GT_STRAW', 0.2), clamp=True)
    gcol = B.mix(straw, gcol, B.rgb('G_TIP', (0.25, 0.25, 0.1)))
    # sopi (temnejse vrzeli med njimi)
    vor = B.node('ShaderNodeTexVoronoi')
    B.set(_sock(vor.inputs, 'Vector'), pos); B.set(_sock(vor.inputs, 'Scale'), B.value('GT_CLUMP', 5.0))
    cl = B.value('GT_CLUMP_AMT', 0.2)
    k2 = B.maprange(_sock(vor.outputs, 'Distance'), 0.15, 0.7, B.math('ADD', 1.0, B.math('MULTIPLY', cl, 0.3)), B.math('SUBTRACT', 1.0, cl))
    gcol = B.mix(1.0, gcol, k2, blend='MULTIPLY')
    # cvetlice (pike)
    fv = B.node('ShaderNodeTexVoronoi')
    B.set(_sock(fv.inputs, 'Vector'), pos); B.set(_sock(fv.inputs, 'Scale'), B.value('FL_SCALE', 30.0))
    frand = B.separate(_sock(fv.outputs, 'Color'))
    dot = B.mul(B.math('LESS_THAN', _sock(fv.outputs, 'Distance'), B.value('FL_SIZE', 0.08)),
                B.math('LESS_THAN', frand[0], B.value('FL_DENS', 0.1)))
    fcol = B.mix(B.math('GREATER_THAN', frand[1], B.value('FL_MIX', 0.5)), B.rgb('H_FY', (0.6, 0.45, 0.02)), B.rgb('H_FW', (0.75, 0.75, 0.7)))
    gcol = B.mix(dot, gcol, fcol)
    under = gcol
    col = B.mix(grassy, soil, under)
    wl = B.value('WL', WATER_LEVEL)
    wet = B.maprange(pz, B.math('ADD', wl, 0.3), B.math('ADD', wl, 0.02), 0.0, 1.0)
    col = B.mix(wet, col, B.rgb('MUD_COL', (0.02, 0.015, 0.008)))
    rough = B.mix(wet, 0.9, 0.65, dtype='FLOAT')
    bn = B.node('ShaderNodeBump')
    B.set(bn.inputs['Strength'], B.value('GT_BUMP', 0.5))
    bn.inputs['Distance'].default_value = 0.02
    hgt = B.mix(grassy, B.noise(pos, 25.0, 4.0),
                B.math('ADD', B.math('MULTIPLY', st1, 0.7), B.math('MULTIPLY', _sock(vor.outputs, 'Distance'), -0.6)), dtype='FLOAT')
    B.set(bn.inputs['Height'], hgt)
    sh = principled(B, col, rough, bn.outputs[0])
    g_veg = th_gray(B, 'veg', 3.0, pos=pos)
    g_veg = B.math('MULTIPLY_ADD', B.math('SUBTRACT', st1, 0.5), B.value('TH_veg_V', 0.03), g_veg)  # drobna tekstura bilk
    g_soil = th_gray(B, 'soil', 2.0, pos=pos)
    g_mud = th_gray(B, 'mud', 2.0, pos=pos)
    g = B.mix(grassy, g_soil, g_veg, dtype='FLOAT')
    g = B.mix(wet, g, g_mud, dtype='FLOAT')
    return finish(m, B, sh, g, 0.0)


def mat_water():
    m, B = new_mat('TD_Water')
    pos = B.out(B.node('ShaderNodeNewGeometry'), 'Position')
    alg = B.maprange(B.noise(pos, 0.25, 3.0), B.value('W_ALGAE', 0.6), B.math('ADD', B.value('W_ALGAE', 0.6), 0.05), 0.0, 1.0)
    col = B.mix(alg, B.rgb('W_COL', (0.01, 0.015, 0.01)), B.rgb('W_ALG_COL', (0.04, 0.07, 0.01)))
    rough = B.mix(alg, B.value('W_ROUGH', 0.05), 0.6, dtype='FLOAT')
    bn = B.node('ShaderNodeBump'); bn.inputs['Strength'].default_value = 0.15
    B.set(bn.inputs['Height'], B.noise(pos, 3.0, 2.0))
    sh = principled(B, col, rough, bn.outputs[0])
    return finish(m, B, sh, th_gray(B, 'water', 0.5), 0.0)


def mat_turtle_shell():
    m, B = new_mat('TD_TurtleShell')
    obj = B.out(B.node('ShaderNodeTexCoord'), 'Object')
    pos = B.out(B.node('ShaderNodeNewGeometry'), 'Position')
    vor = B.node('ShaderNodeTexVoronoi', feature='DISTANCE_TO_EDGE')
    B.set(_sock(vor.inputs, 'Vector'), obj); B.set(_sock(vor.inputs, 'Scale'), 3.2)
    seam = B.maprange(_sock(vor.outputs, 'Distance'), 0.0, 0.05, 1.0, 0.0)
    base = B.rgb('TU_SHELL', (0.02, 0.02, 0.01))
    col = B.mix(B.mul(seam, B.value('TU_SEAM', 0.7)), base, B.rgb('TU_SEAM_COL', (0.005, 0.005, 0.003)))
    sp = B.noise(obj, 14.0, 2.0)
    spot = B.maprange(sp, B.value('TU_SPECK_THR', 0.68), B.math('ADD', B.value('TU_SPECK_THR', 0.68), 0.03), 0.0, 1.0)
    col = B.mix(spot, col, B.rgb('TU_SPOT', (0.35, 0.28, 0.04)))
    dirt = B.mul(B.maprange(B.noise(pos, 20.0, 3.0), 0.45, 0.65, 0.0, 1.0), B.value('TU_DIRT', 0.2))
    col = B.mix(dirt, col, B.rgb('S_COL', (0.07, 0.045, 0.02)))
    bn = B.node('ShaderNodeBump'); bn.inputs['Strength'].default_value = 0.4
    B.set(bn.inputs['Height'], B.math('ADD', B.mul(seam, -1.0), B.mul(B.noise(obj, 30.0, 4.0), 0.3)))
    rough = B.mix(dirt, B.value('TU_ROUGH', 0.35), 0.85, dtype='FLOAT')
    sh = principled(B, col, rough, bn.outputs[0])
    return finish(m, B, sh, th_gray(B, 'turtle', 15.0, pos=obj), 1.0)


def mat_turtle_skin():
    m, B = new_mat('TD_TurtleSkin')
    obj = B.out(B.node('ShaderNodeTexCoord'), 'Object')
    thr = B.math('ADD', B.value('TU_SPECK_THR', 0.68), 0.06)
    sp = B.maprange(B.noise(obj, 40.0, 2.0), thr, B.math('ADD', thr, 0.03), 0.0, 1.0)
    col = B.mix(sp, B.rgb('TU_SKIN', (0.03, 0.03, 0.02)), B.rgb('TU_SPOT', (0.35, 0.28, 0.04)))
    sh = principled(B, col, 0.45)
    return finish(m, B, sh, th_gray(B, 'turtleskin', 15.0, pos=obj), 1.0)


def make_world():
    w = bpy.data.worlds.new('TD_World')
    try:
        w.use_nodes = True
    except Exception:
        pass
    w.node_tree.nodes.clear()
    B = NT(w.node_tree)
    z = B.separate(B.out(B.node('ShaderNodeTexCoord'), 'Generated'))[2]
    col = B.mix(B.maprange(z, 0.0, 0.6, 0.0, 1.0), B.rgb('SKY_HOR', (0.7, 0.8, 0.9)), B.rgb('SKY_ZEN', (0.2, 0.35, 0.7)))
    col = B.mix(B.maprange(z, -0.02, 0.02, 1.0, 0.0), col, B.rgb('SKY_GND', (0.05, 0.06, 0.03)))
    mt, mm = B.value('MODE_T', 0.0), B.value('MODE_M', 0.0)
    skyc = B.node('ShaderNodeCombineColor')
    B.set(skyc.inputs[0], B.value('TH_sky_B', 0.0)); B.set(skyc.inputs[1], B.value('TH_sky_B', 0.0))
    skyc.inputs[2].default_value = 0.0
    col = B.mix(mt, col, skyc.outputs[0])
    col = B.mix(mm, col, (0.0, 0.0, 0.0))
    strength = B.mix(mt, B.value('W_STR', 1.0), 1.0, dtype='FLOAT')
    bg = B.node('ShaderNodeBackground'); B.set(bg.inputs['Color'], col); B.set(bg.inputs['Strength'], strength)
    o = B.node('ShaderNodeOutputWorld'); B.L.new(bg.outputs[0], o.inputs['Surface'])
    return w


# ---------------------------------------------------------------- geometry helpers
def mesh_obj(name, bm, mats, coll, link=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    for mt in mats:
        me.materials.append(mt)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    if link:
        coll.objects.link(ob)
    return ob


def set_mat_idx(verts, idx):
    for f in {f for v in verts for f in v.link_faces}:
        f.material_index = idx


def add_blade(bm, w, bend, h=1.0, seg=5, M=None, idx=0, curl=0.0):
    M = M or Matrix.Identity(4)
    rows = []
    for i in range(seg + 1):
        t = i / seg
        ww = w * (1 - t) ** 0.6
        y = bend * h * t * t
        z = h * t * (1 - 0.35 * bend * t)
        if i < seg:
            rows.append([bm.verts.new(M @ Vector((-ww / 2, y - curl * ww, z))),
                         bm.verts.new(M @ Vector((0, y, z))),
                         bm.verts.new(M @ Vector((ww / 2, y - curl * ww, z)))])
        else:
            rows.append([bm.verts.new(M @ Vector((0, y, z)))])
    for i in range(seg - 1):
        a, b = rows[i], rows[i + 1]
        for k in range(2):
            f = bm.faces.new((a[k], a[k + 1], b[k + 1], b[k])); f.material_index = idx
    a, tip = rows[seg - 1], rows[seg][0]
    for k in range(2):
        f = bm.faces.new((a[k], a[k + 1], tip)); f.material_index = idx


def add_stem(bm, h, r=0.0015, M=None, idx=0):
    M = (M or Matrix.Identity(4)) @ Matrix.Translation((0, 0, h / 2))
    res = bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, segments=4,
                                radius1=r, radius2=r * 0.7, depth=h, matrix=M)
    set_mat_idx(res['verts'], idx)


def add_sphere(bm, r, M, idx, u=8, v=6):
    res = bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r, matrix=M)
    set_mat_idx(res['verts'], idx)
    return res['verts']


def add_disc(bm, r, M, idx, seg=8):
    res = bmesh.ops.create_circle(bm, cap_ends=True, segments=seg, radius=r, matrix=M)
    set_mat_idx(res['verts'], idx)


def TRS(t=(0, 0, 0), r=(0, 0, 0), s=(1, 1, 1)):
    return Matrix.Translation(t) @ Euler(r).to_matrix().to_4x4() @ Matrix.Diagonal((*s, 1))


# ---------------------------------------------------------------- templates (vegetation)
def build_templates(coll, M, rnd):
    grass, herbs, reeds = [], [], []
    # sopi trave (vsak 8-30 bilk) - redko posejani cez teksturo
    for i, (n, spread) in enumerate([(8, 0.02), (14, 0.035), (20, 0.05), (30, 0.07), (12, 0.03)]):
        bm = bmesh.new()
        for k in range(n):
            a = rnd.uniform(0, 2 * math.pi); d = spread * math.sqrt(rnd.random())
            add_blade(bm, rnd.uniform(0.003, 0.007), rnd.uniform(0.05, 0.5), h=rnd.uniform(0.5, 1.0), seg=4,
                      M=TRS((d * math.cos(a), d * math.sin(a), 0), (rnd.uniform(-0.35, 0.1), 0, a + rnd.uniform(-0.5, 0.5))),
                      curl=rnd.uniform(0, 0.4))
        grass.append(mesh_obj(f'TPL_tuft{i}', bm, [M['grass']], coll))
    # detelja
    for i in range(2):
        bm = bmesh.new()
        for j in range(3 + i):
            h = 0.04 + 0.03 * j
            add_stem(bm, h, 0.0008, TRS((0.01 * j, -0.008 * j, 0)), 0)
            for k in range(3):
                a = k * 2.094
                add_disc(bm, 0.009, TRS((0.01 * j + 0.009 * math.cos(a), -0.008 * j + 0.009 * math.sin(a), h), (0.2, 0, a)), 0)
        herbs.append(mesh_obj(f'TPL_clover{i}', bm, [M['leaf']], coll))
    # trpotec (rozeta)
    bm = bmesh.new()
    for k in range(6):
        a = k * 1.047 + 0.3
        add_blade(bm, 0.035, 0.0, h=0.11, seg=4, M=TRS((0, 0, 0.003), (1.25, 0, a)))
    herbs.append(mesh_obj('TPL_plantain', bm, [M['leaf']], coll))
    # regrat, marjetica, zlatica, latovka (semena)
    for name, fm, h, kind in [('dandelion', 'flower_y', 0.18, 'ball'), ('daisy', 'flower_w', 0.08, 'disc'),
                              ('buttercup', 'flower_y', 0.3, 'cup'), ('clover_flower', 'flower_p', 0.12, 'ball'),
                              ('seedhead', 'seed', 0.55, 'ear')]:
        bm = bmesh.new()
        add_stem(bm, h, 0.0015, None, 0)
        if kind == 'ball':
            add_sphere(bm, 0.009, TRS((0, 0, h)), 1)
        elif kind == 'disc':
            add_disc(bm, 0.01, TRS((0, 0, h), (0.15, 0, 0)), 1, 12)
            add_sphere(bm, 0.004, TRS((0, 0, h + 0.002)), 2)
        elif kind == 'cup':
            add_sphere(bm, 0.01, TRS((0, 0, h), s=(1, 1, 0.6)), 1)
        else:
            add_sphere(bm, 0.008, TRS((0, 0, h), s=(1, 1, 5)), 1)
        if kind != 'ear':
            for k in range(4):
                add_blade(bm, 0.02, 0.1, h=0.07, seg=3, M=TRS((0, 0, 0.002), (1.1, 0, k * 1.57)), idx=3)
        mats = [M['grass'], M[fm], M['flower_y'], M['leaf']]
        herbs.append(mesh_obj(f'TPL_{name}', bm, mats, coll))
    # trst / rogoz
    for i, (w, bend) in enumerate([(0.012, 0.1), (0.015, 0.25), (0.01, 0.05)]):
        bm = bmesh.new(); add_blade(bm, w, bend, seg=6, curl=0.3)
        add_blade(bm, w * 0.8, bend * 1.5, h=0.8, seg=6, M=TRS((0.01, 0, 0), (0, 0, 2.5)))
        reeds.append(mesh_obj(f'TPL_reed{i}', bm, [M['reed']], coll))
    bm = bmesh.new(); add_stem(bm, 1.0, 0.004, None, 0)
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.012, radius2=0.012, depth=0.15,
                                matrix=Matrix.Translation((0, 0, 0.8)))
    set_mat_idx(res['verts'], 1)
    reeds.append(mesh_obj('TPL_cattail', bm, [M['reed'], M['bark']], coll))
    return grass, herbs, reeds


def build_veg_group(c_grass, c_herb, c_reed):
    ng = bpy.data.node_groups.new('TD_Vegetation', 'GeometryNodeTree')
    ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    B = NT(ng)
    gi, go = B.node('NodeGroupInput'), B.node('NodeGroupOutput')
    geo = gi.outputs[0]
    pos = B.node('GeometryNodeInputPosition').outputs[0]
    px, py, pz = B.separate(pos)

    def d2(cx, cy):
        dx, dy = B.math('SUBTRACT', px, cx), B.math('SUBTRACT', py, cy)
        return B.math('SQRT', B.math('ADD', B.math('MULTIPLY', dx, dx), B.math('MULTIPLY', dy, dy)))

    R = B.value('R', 20.0)
    dc = d2(B.value('CX', 0.0), B.value('CY', 0.0))
    in_patch = B.math('LESS_THAN', dc, R)
    fall = B.maprange(d2(B.value('CAMX', 0.0), B.value('CAMY', 0.0)), B.value('NEAR_D', 6.0), B.value('FAR_D', 25.0),
                      1.0, B.value('FAR', 0.0))
    nz = B.noise(pos, B.value('N_SCALE', 0.15), 2.0, w=B.value('N_W', 0.0))
    thr = B.value('BARE_THR', 0.3)
    grassy = B.maprange(nz, B.math('SUBTRACT', thr, 0.04), B.math('ADD', thr, 0.04), 0.0, 1.0)
    excl = B.math('GREATER_THAN', d2(B.value('TX', 0.0), B.value('TY', 0.0)), B.value('TR', 0.1))
    wl = B.value('WL', WATER_LEVEL)
    dryland = B.math('GREATER_THAN', pz, B.math('ADD', wl, 0.03))
    mask = B.mul(in_patch, grassy, excl, dryland)
    seed = B.value('SEED', 0.0)

    def distribute(density, k):
        d = B.node('GeometryNodeDistributePointsOnFaces', distribute_method='RANDOM')
        B.set(_sock(d.inputs, 'Mesh'), geo); B.set(_sock(d.inputs, 'Density'), density)
        B.set(_sock(d.inputs, 'Seed'), B.math('ADD', seed, k))
        return _sock(d.outputs, 'Points')

    def rnd(dtype, lo, hi, k):
        n = B.node('FunctionNodeRandomValue', data_type=dtype)
        B.set(_sock(n.inputs, 'Min'), lo); B.set(_sock(n.inputs, 'Max'), hi)
        B.set(_sock(n.inputs, 'Seed'), B.math('ADD', seed, k))
        return _sock(n.outputs, 'Value')

    def e2r(v):
        n = B.node('FunctionNodeEulerToRotation'); B.set(n.inputs[0], v); return n.outputs[0]

    def instance(points, coll, n_items, rot, scale, k):
        ci = B.node('GeometryNodeCollectionInfo', transform_space='ORIGINAL')
        _sock(ci.inputs, 'Collection').default_value = coll
        _sock(ci.inputs, 'Separate Children').default_value = True
        _sock(ci.inputs, 'Reset Children').default_value = True
        io = B.node('GeometryNodeInstanceOnPoints')
        B.set(_sock(io.inputs, 'Points'), points); B.set(_sock(io.inputs, 'Instance'), ci.outputs[0])
        _sock(io.inputs, 'Pick Instance').default_value = True
        B.set(_sock(io.inputs, 'Instance Index'), rnd('INT', 0, n_items - 1, k))
        B.set(_sock(io.inputs, 'Rotation'), rot); B.set(_sock(io.inputs, 'Scale'), scale)
        return io.outputs[0]

    # trava
    tilt = B.value('TILT', 0.25); ntilt = B.math('MULTIPLY', tilt, -1.0)
    rot = e2r(B.combine(rnd('FLOAT', ntilt, tilt, 11), rnd('FLOAT', ntilt, tilt, 12), rnd('FLOAT', 0.0, 6.2832, 13)))
    lean = e2r(B.combine(B.value('LEAN_X', 0.0), B.value('LEAN_Y', 0.0), 0.0))
    try:
        rr = B.node('FunctionNodeRotateRotation')
        rr.rotation_space = 'GLOBAL'
        B.set(rr.inputs[0], rot); B.set(rr.inputs[1], lean); rot = rr.outputs[0]
    except Exception as ex:
        print('RotateRotation ni na voljo:', ex)
    hv = B.value('HEIGHT_VAR', 0.3)
    hmod = B.maprange(B.noise(pos, B.value('N2_SCALE', 0.3), 2.0, w=B.math('ADD', B.value('N_W', 0.0), 13.0)),
                      0.3, 0.7, B.value('MOW_MIN', 0.6), 1.0)
    h = B.mul(B.value('HEIGHT', 0.15), rnd('FLOAT', B.math('SUBTRACT', 1.0, hv), B.math('ADD', 1.0, hv), 14), hmod)
    w = rnd('FLOAT', 0.7, 1.4, 15)
    pts = distribute(B.mul(mask, fall, B.value('DENS_G', 500.0)), 0)
    g_inst = instance(pts, c_grass, len(c_grass.objects), rot, B.combine(w, w, h), 16)
    # zelisca / cvetlice
    pts_h = distribute(B.mul(mask, fall, B.value('DENS_H', 10.0)), 1)
    rot_h = e2r(B.combine(rnd('FLOAT', -0.15, 0.15, 21), rnd('FLOAT', -0.15, 0.15, 22), rnd('FLOAT', 0.0, 6.2832, 23)))
    hs = B.mul(rnd('FLOAT', 0.6, 1.4, 24), B.value('HERB_SCALE', 1.0))
    h_inst = instance(pts_h, c_herb, len(c_herb.objects), rot_h, B.combine(hs, hs, hs), 25)
    # trst ob mlaki
    band = B.mul(B.math('GREATER_THAN', pz, B.math('SUBTRACT', wl, 0.25)),
                 B.math('LESS_THAN', pz, B.math('ADD', wl, 0.2)),
                 B.math('LESS_THAN', dc, B.math('ADD', R, 10.0)))
    pts_r = distribute(B.mul(band, B.value('DENS_R', 40.0)), 2)
    rot_r = e2r(B.combine(rnd('FLOAT', -0.2, 0.2, 31), rnd('FLOAT', -0.2, 0.2, 32), rnd('FLOAT', 0.0, 6.2832, 33)))
    rh = B.mul(B.value('REED_H', 1.3), rnd('FLOAT', 0.6, 1.3, 34))
    rw = rnd('FLOAT', 0.8, 1.3, 35)
    r_inst = instance(pts_r, c_reed, len(c_reed.objects), rot_r, B.combine(rw, rw, rh), 36)

    j = B.node('GeometryNodeJoinGeometry')
    for s in (geo, g_inst, h_inst, r_inst):
        B.L.new(s, j.inputs[0])
    B.L.new(j.outputs[0], go.inputs[0])
    return ng


# ---------------------------------------------------------------- scene pieces
def build_terrain(coll, mat):
    hx, hy = MEADOW_X / 2, MEADOW_Y / 2
    nx, ny = int(MEADOW_X / GRID), int(MEADOW_Y / GRID)
    verts = [(-hx + i * GRID, -hy + j * GRID, _raw_height(-hx + i * GRID, -hy + j * GRID))
             for j in range(ny + 1) for i in range(nx + 1)]
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i)
             for j in range(ny) for i in range(nx)]
    me = bpy.data.meshes.new('TD_Terrain'); me.from_pydata(verts, [], faces); me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.new('TD_Terrain', me); coll.objects.link(ob)
    return ob


def build_ring(name, coll, mat, ix, iy, ox, oy, step, z=0.0):
    """Pravokotni 'obroc' okoli travnika (zunanja tla)."""
    xs = sorted(set([-ox + k * step for k in range(int(2 * ox / step) + 1)] + [-ix, ix]))
    ys = sorted(set([-oy + k * step for k in range(int(2 * oy / step) + 1)] + [-iy, iy]))
    vid, verts, faces = {}, [], []
    def v(i, j):
        if (i, j) not in vid:
            vid[(i, j)] = len(verts); verts.append((xs[i], ys[j], z))
        return vid[(i, j)]
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if abs(cx) < ix and abs(cy) < iy:
                continue
            faces.append((v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)))
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    return ob


def build_tree_templates(M, rnd, coll):
    temps = []
    for t in range(6):
        bm = bmesh.new()
        conifer = t >= 4
        th = rnd.uniform(2.0, 4.0) if not conifer else rnd.uniform(1.0, 2.0)
        tr = rnd.uniform(0.15, 0.3)
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=tr, radius2=tr * 0.6,
                                    depth=th + 1.0, matrix=Matrix.Translation((0, 0, (th + 1.0) / 2 - 0.3)))
        set_mat_idx(res['verts'], 0)
        if conifer:
            H = rnd.uniform(10, 18); n = 5
            for k in range(n):
                f = k / n
                r = (1 - f) * rnd.uniform(2.2, 3.0) + 0.3
                res = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=r, radius2=0.05,
                                            depth=H / n * 1.8, matrix=Matrix.Translation((0, 0, th + f * H + H / n * 0.9)))
                set_mat_idx(res['verts'], 1)
                for vv in res['verts']:
                    vv.co += Vector((rnd.uniform(-.2, .2), rnd.uniform(-.2, .2), rnd.uniform(-.2, .2)))
            crad, cz, ctop = 2.8, th + H * 0.4, th + H
        else:
            crad = rnd.uniform(2.5, 4.5); cz = th + crad * 0.9
            for k in range(rnd.randint(6, 10)):
                r = crad * rnd.uniform(0.45, 0.75)
                off = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.6, 0.8))) * crad * 0.55
                res = bmesh.ops.create_icosphere(bm, subdivisions=3, radius=r, matrix=Matrix.Translation(Vector((0, 0, cz)) + off))
                set_mat_idx(res['verts'], 1)
                for vv in res['verts']:
                    d = (vv.co - Vector((0, 0, cz)) - off).normalized()
                    vv.co += d * r * 0.3 * noise.noise(vv.co * 0.9 + Vector((t * 10, k, 0)))
            ctop = cz + crad
        ob = mesh_obj(f'TPL_tree{t}', bm, [M['bark'], M['leaves']], coll)
        ob['canopy_r'], ob['canopy_z'], ob['top'] = crad, cz, ctop
        temps.append(ob)
    return temps


def place_trees(temps, rnd, coll):
    hx, hy = MEADOW_X / 2, MEADOW_Y / 2
    trees = []
    def add(x, y):
        t = rnd.choice(temps)
        s = rnd.uniform(0.75, 1.25)
        ob = bpy.data.objects.new('TD_Tree', t.data); coll.objects.link(ob)
        ob.location = (x, y, -0.1); ob.rotation_euler = (0, 0, rnd.uniform(0, 6.28)); ob.scale = (s, s, s)
        ob['canopy_r'], ob['canopy_z'], ob['top'] = t['canopy_r'] * s, t['canopy_z'] * s, t['top'] * s
        trees.append(ob)
    for (lo, hi, gap) in [(1.5, 6.0, (4.0, 9.0)), (7.0, 25.0, (3.0, 6.0))]:
        per = 2 * (MEADOW_X + MEADOW_Y) + 8 * hi
        s = 0.0
        while s < per:
            s += rnd.uniform(*gap)
            u = s
            off = rnd.uniform(lo, hi)
            W, Hh = MEADOW_X + 2 * off, MEADOW_Y + 2 * off
            u = u % (2 * (W + Hh))
            if u < W: x, y = -W / 2 + u, -Hh / 2
            elif u < W + Hh: x, y = W / 2, -Hh / 2 + (u - W)
            elif u < 2 * W + Hh: x, y = W / 2 - (u - W - Hh), Hh / 2
            else: x, y = -W / 2, Hh / 2 - (u - 2 * W - Hh)
            if rnd.random() < 0.15 and lo < 2:
                continue  # vrzeli v drevoredu
            add(x, y)
    return trees


def build_turtle(M, coll):
    root = bpy.data.objects.new('Turtle', None); coll.objects.link(root)
    root.empty_display_size = 0.12
    parts = {}
    bm = bmesh.new()
    res = bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=1.0)
    for v in res['verts']:
        x, y, z = v.co
        ridge = 1.0 + 0.08 * math.exp(-(y * y) / 0.02)
        flare = 1.0 + 0.06 * max(0.0, -x) ** 2
        z = z * 0.048 * ridge if z > 0 else z * 0.012
        v.co = Vector((x * 0.10, y * 0.075 * flare, z + 0.022))
    parts['shell'] = mesh_obj('Turtle_Shell', bm, [M['shell']], coll)
    bm = bmesh.new(); add_sphere(bm, 1.0, TRS(s=(0.03, 0.021, 0.018)), 0, 16, 10)
    parts['head'] = mesh_obj('Turtle_Head', bm, [M['skin']], coll)
    for nm, (x, y) in {'leg_fl': (0.06, 0.06), 'leg_fr': (0.06, -0.06), 'leg_bl': (-0.06, 0.055), 'leg_br': (-0.06, -0.055)}.items():
        bm = bmesh.new(); add_sphere(bm, 1.0, TRS(s=(0.03, 0.017, 0.011)), 0, 12, 8)
        ob = mesh_obj('Turtle_' + nm, bm, [M['skin']], coll); ob['base'] = (x, y); parts[nm] = ob
    bm = bmesh.new()
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.008, radius2=0.001, depth=0.04,
                                matrix=TRS((0, 0, 0), (0, -math.pi / 2, 0)))
    parts['tail'] = mesh_obj('Turtle_Tail', bm, [M['skin']], coll)
    for p in parts.values():
        p.parent = root
    return root, parts


def pose_turtle(parts, r):
    out = r.uniform(0, 1)  # glava/noge: 0 = skrita v oklepu, 1 = iztegnjena
    parts['head'].location = (0.075 + 0.045 * out, 0, 0.02 + r.uniform(-0.004, 0.008))
    parts['head'].rotation_euler = (0, r.uniform(-0.2, 0.2), r.uniform(-0.5, 0.5) * out)
    parts['head'].scale = [0.6 + 0.4 * out] * 3
    for nm in ('leg_fl', 'leg_fr', 'leg_bl', 'leg_br'):
        x, y = parts[nm]['base']
        o = clamp(out + r.uniform(-0.3, 0.3))
        parts[nm].location = (x * (0.9 + 0.25 * o), y * (0.85 + 0.35 * o), 0.01)
        parts[nm].rotation_euler = (0, 0, math.copysign(0.5, x * y) + r.uniform(-0.4, 0.4))
        parts[nm].scale = [0.6 + 0.4 * o] * 3
    parts['tail'].location = (-0.1 - 0.01 * out, 0, 0.012)


def build_props(M, rnd, coll):
    props = []
    for i in range(30):
        bm = bmesh.new()
        res = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        sd = Vector((rnd.uniform(0, 50), rnd.uniform(0, 50), 0))
        for v in res['verts']:
            v.co *= 1.0 + 0.3 * noise.noise(v.co * 1.3 + sd)
        ob = mesh_obj(f'TD_Stone{i}', bm, [M['stone']], coll)
        s = rnd.uniform(0.04, 0.25)
        ob.scale = (s * rnd.uniform(0.8, 1.5), s, s * rnd.uniform(0.35, 0.8))
        ob['kind'] = 'stone'; props.append(ob)
    for i in range(12):
        bm = bmesh.new()
        res = bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
        sd = Vector((rnd.uniform(0, 50), 0, 0))
        for v in res['verts']:
            v.co *= 1.0 + 0.15 * noise.noise(v.co * 3 + sd)
        ob = mesh_obj(f'TD_Molehill{i}', bm, [M['mole']], coll)
        s = rnd.uniform(0.15, 0.3); ob.scale = (s, s * rnd.uniform(0.8, 1.2), s * rnd.uniform(0.4, 0.7))
        ob['kind'] = 'mole'; props.append(ob)
    for i in range(10):
        bm = bmesh.new()
        L = rnd.uniform(0.2, 1.0); rr = rnd.uniform(0.006, 0.025)
        bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=rr, radius2=rr * 0.6, depth=L,
                              matrix=TRS((0, 0, 0), (0, math.pi / 2, 0)))
        ob = mesh_obj(f'TD_Stick{i}', bm, [M['bark']], coll)
        ob['kind'] = 'stick'; props.append(ob)
    for ob in props:
        ob.hide_render = True; ob.location = (0, 0, -50)
    return props


def build_scene(seed=1):
    rnd = random.Random(seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = 'TurtleMeadow'
    root = scene.collection
    def mkcoll(n):
        c = bpy.data.collections.new(n); root.children.link(c); return c
    c_env, c_tpl, c_trees, c_props, c_turtle = (mkcoll(n) for n in ('TD_Environment', 'TD_Templates', 'TD_Trees', 'TD_Props', 'TD_Turtle'))
    c_tg, c_th, c_tr = (bpy.data.collections.new(n) for n in ('TD_TplGrass', 'TD_TplHerb', 'TD_TplReed'))
    for c in (c_tg, c_th, c_tr):
        c_tpl.children.link(c)

    M = {
        'grass': mat_grass(),
        'ground': mat_ground(),
        'water': mat_water(),
        'shell': mat_turtle_shell(),
        'skin': mat_turtle_skin(),
        'leaf': mat_simple('TD_Leaf', 'H_LEAF', (0.04, 0.09, 0.02), 'veg', 0.5, 0.25, 0.25),
        'flower_y': mat_simple('TD_FlowerY', 'H_FY', (0.6, 0.45, 0.02), 'veg', 0.6, 0.2, 0.15),
        'flower_w': mat_simple('TD_FlowerW', 'H_FW', (0.75, 0.75, 0.7), 'veg', 0.6, 0.2, 0.1),
        'flower_p': mat_simple('TD_FlowerP', 'H_FP', (0.35, 0.12, 0.3), 'veg', 0.6, 0.2, 0.15),
        'seed': mat_simple('TD_Seed', 'H_SEED', (0.15, 0.1, 0.06), 'veg', 0.7, 0.2, 0.3),
        'reed': mat_simple('TD_Reed', 'R_COL', (0.06, 0.1, 0.03), 'veg', 0.5, 0.3, 0.3, 1.0),
        'bark': mat_simple('TD_Bark', 'BARK_COL', (0.04, 0.03, 0.02), 'tree', 0.85, 0.0, 0.3, 6.0, bump=0.5),
        'leaves': mat_simple('TD_Leaves', 'T_COL', (0.03, 0.07, 0.02), 'tree', 0.6, 0.3, 0.25, 3.0, bump=0.6),
        'stone': mat_simple('TD_Stone', 'ST_COL', (0.15, 0.14, 0.12), 'stone', 0.7, 0.0, 0.45, 15.0, bump=0.4),
        'mole': mat_simple('TD_Molehill', 'S_COL', (0.07, 0.045, 0.02), 'mole', 0.9, 0.0, 0.2, 30.0, bump=0.8),
    }
    gtpl, htpl, rtpl = build_templates(c_tg, M, rnd)
    for ob in htpl:
        c_tg.objects.unlink(ob); c_th.objects.link(ob)
    for ob in rtpl:
        c_tg.objects.unlink(ob); c_tr.objects.link(ob)
    ttpl = build_tree_templates(M, rnd, c_tpl)

    terrain = build_terrain(c_env, M['ground'])
    ring = build_ring('TD_GroundNear', c_env, M['ground'], MEADOW_X / 2, MEADOW_Y / 2, 90, 65, 1.0)
    far = build_ring('TD_GroundFar', c_env, M['ground'], 90, 65, 1500, 1500, 30.0, z=0.0)
    ng = build_veg_group(c_tg, c_th, c_tr)
    for ob in (terrain, ring):
        mod = ob.modifiers.new('Vegetation', 'NODES'); mod.node_group = ng

    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    water = mesh_obj('TD_Water', bm, [M['water']], c_env)
    water.location = (POND['cx'], POND['cy'], WATER_LEVEL)
    water.scale = (POND['rx'] * 1.5, POND['ry'] * 1.5, 1)

    place_trees(ttpl, rnd, c_trees)
    build_props(M, rnd, c_props)
    build_turtle(M, c_turtle)

    bpy.context.view_layer.layer_collection.children['TD_Templates'].exclude = True
    for ob in ttpl:  # predloge dreves so v izkljuceni zbirki; drevesa so povezane kopije
        pass

    scene.world = make_world()
    sun_data = bpy.data.lights.new('TD_Sun', 'SUN')
    sun = bpy.data.objects.new('TD_Sun', sun_data); root.objects.link(sun)
    cam_data = bpy.data.cameras.new('TD_Camera')
    cam_data.sensor_fit = 'HORIZONTAL'; cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.05; cam_data.clip_end = 3000
    cam = bpy.data.objects.new('TD_Camera', cam_data); root.objects.link(cam)
    scene.camera = cam

    setup_render(scene, use_cycles=False)
    # skripta v .blend
    try:
        src = open(os.path.abspath(__file__), encoding='utf-8').read()
        txt = bpy.data.texts.new('turtle_dataset.py'); txt.from_string(src)
    except Exception:
        pass
    return scene


# ---------------------------------------------------------------- render setup
def setup_render(scene, use_cycles=False):
    if use_cycles:
        scene.render.engine = 'CYCLES'
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            for dt in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
                try:
                    prefs.compute_device_type = dt; prefs.get_devices()
                    for d in prefs.devices: d.use = True
                    scene.cycles.device = 'GPU'; break
                except Exception:
                    continue
        except Exception:
            pass
        scene.cycles.use_denoising = True
    else:
        for eng in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
            try:
                scene.render.engine = eng; break
            except TypeError:
                continue
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'


def set_samples(scene, n):
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = max(4, n * 4)
    else:
        scene.eevee.taa_render_samples = n


# ---------------------------------------------------------------- param setting
_PARAM_CACHE = {}

def _build_cache():
    _PARAM_CACHE.clear()
    trees = [m.node_tree for m in bpy.data.materials if m.node_tree] + \
            [w.node_tree for w in bpy.data.worlds if w.node_tree] + list(bpy.data.node_groups)
    for t in trees:
        for n in t.nodes:
            if n.type in ('VALUE', 'RGB'):
                _PARAM_CACHE.setdefault(n.name, []).append(n)


def set_val(name, v):
    if not _PARAM_CACHE:
        _build_cache()
    for n in _PARAM_CACHE.get(name, []):
        if n.type == 'VALUE':
            n.outputs[0].default_value = float(v)
        else:
            n.outputs[0].default_value = (*v[:3], 1.0)


def set_view(scene, transform, exposure=0.0, look=None):
    vs = scene.view_settings
    for t in ([transform] if isinstance(transform, str) else transform):
        try:
            vs.view_transform = t; break
        except TypeError:
            continue
    looks = [look] if look else []
    for lk in looks + ['None']:
        try:
            vs.look = lk; break
        except TypeError:
            continue
    vs.exposure = exposure; vs.gamma = 1.0


# ---------------------------------------------------------------- per-sample randomization
def randomize_look(r, scene):
    P = {}
    regime = r.random()
    if regime < 0.3:   P['grass_h'] = U(r, 0.04, 0.10)    # pokosen
    elif regime < 0.75: P['grass_h'] = U(r, 0.10, 0.25)
    else:              P['grass_h'] = U(r, 0.25, 0.50)    # visoka trava
    season = r.random()   # 0 = spomladi sveze, 1 = poletje suho
    gh = U(r, 0.19, 0.30) - 0.06 * season
    gA = hsv(gh, U(r, 0.5, 0.85), U(r, 0.07, 0.18))
    gB = hsv(gh + U(r, -0.04, 0.04), U(r, 0.45, 0.85), U(r, 0.07, 0.2))
    dry = hsv(U(r, 0.08, 0.13), U(r, 0.3, 0.6), U(r, 0.12, 0.35))
    soil = hsv(U(r, 0.04, 0.09), U(r, 0.3, 0.65), U(r, 0.02, 0.10))
    vals = dict(
        G_COL_A=gA, G_COL_B=gB, G_DRY=dry, G_TIP=hsv(U(r, 0.12, 0.2), U(r, 0.3, 0.6), U(r, 0.1, 0.3)),
        G_NSCALE=U(r, 0.08, 0.6), G_DSCALE=U(r, 0.05, 0.5), G_DRY_FRAC=clamp(season * U(r, 0.1, 0.6) + U(r, 0, 0.08)),
        G_HUEJ=U(r, 0.01, 0.06), G_VALJ=U(r, 0.2, 0.7), G_TIP_AMT=U(r, 0.0, 0.6), G_BASE_DARK=U(r, 0.3, 0.7),
        S_COL=soil, S_COL_D=scl(soil, U(r, 0.4, 0.7)), MUD_COL=scl(soil, U(r, 0.25, 0.45)),
        U_COL_A=scl(hsv(gh - 0.03, 0.6, 1), U(r, 0.035, 0.08)), U_COL_B=hsv(U(r, 0.07, 0.12), U(r, 0.35, 0.55), U(r, 0.03, 0.08)),
        N_SCALE=U(r, 0.05, 0.35), N_W=U(r, 0, 100), BARE_THR=(0.0 if r.random() < 0.3 else U(r, 0.2, 0.42)),
        W_COL=hsv(U(r, 0.1, 0.3), U(r, 0.3, 0.7), U(r, 0.008, 0.035)), W_ALG_COL=hsv(U(r, 0.2, 0.28), 0.7, U(r, 0.03, 0.08)),
        W_ALGAE=U(r, 0.55, 0.85), W_ROUGH=U(r, 0.15, 0.4),
        H_LEAF=scl(gA, U(r, 0.8, 1.3)), H_FY=hsv(U(r, 0.11, 0.15), U(r, 0.8, 1), U(r, 0.4, 0.7)),
        H_FW=hsv(0.15, U(r, 0, 0.1), U(r, 0.6, 0.85)), H_FP=hsv(U(r, 0.8, 0.95), U(r, 0.4, 0.7), U(r, 0.15, 0.4)),
        H_SEED=hsv(U(r, 0.05, 0.12), U(r, 0.3, 0.6), U(r, 0.08, 0.25)), R_COL=hsv(gh, U(r, 0.4, 0.8), U(r, 0.04, 0.12)),
        BARK_COL=hsv(U(r, 0.05, 0.1), U(r, 0.2, 0.5), U(r, 0.02, 0.06)),
        T_COL=(hsv(U(r, 0.05, 0.12), U(r, 0.6, 0.9), U(r, 0.05, 0.15)) if r.random() < 0.12 else
               hsv(U(r, 0.2, 0.32), U(r, 0.5, 0.85), U(r, 0.025, 0.08))),
        ST_COL=hsv(U(r, 0.05, 0.15), U(r, 0.05, 0.35), U(r, 0.04, 0.16)),
    )
    # zelva
    kind = r.random()
    if kind < 0.5:   shell = hsv(U(r, 0.1, 0.17), U(r, 0.3, 0.6), U(r, 0.008, 0.025))  # temna (mocvirska sklednica)
    elif kind < 0.85: shell = hsv(U(r, 0.08, 0.16), U(r, 0.4, 0.75), U(r, 0.02, 0.07))
    else:            shell = hsv(U(r, 0.06, 0.12), U(r, 0.4, 0.7), U(r, 0.06, 0.14))
    vals.update(TU_SHELL=shell, TU_SEAM_COL=scl(shell, 0.3), TU_SEAM=U(r, 0.3, 0.9),
                TU_SPOT=hsv(U(r, 0.11, 0.15), U(r, 0.6, 0.95), U(r, 0.15, 0.55)),
                TU_SPECK_THR=U(r, 0.62, 0.85), TU_DIRT=U(r, 0, 0.7), TU_ROUGH=U(r, 0.35, 0.75),
                TU_SKIN=hsv(U(r, 0.08, 0.2), U(r, 0.2, 0.5), U(r, 0.01, 0.05)))
    # nebo in sonce
    overcast = r.random() < 0.25
    vals.update(SKY_HOR=hsv(0.58, U(r, 0.05, 0.25), U(r, 0.6, 1.0)),
                SKY_ZEN=hsv(U(r, 0.57, 0.63), (U(r, 0.0, 0.15) if overcast else U(r, 0.35, 0.7)), U(r, 0.35, 0.8)),
                SKY_GND=scl(gA, 0.8), W_STR=(U(r, 1.2, 2.2) if overcast else U(r, 0.8, 1.5)))
    P['sun'] = dict(energy=(U(r, 0.2, 0.8) if overcast else U(r, 2.5, 5.0)),
                    elev=U(r, 15, 75), azim=U(r, 0, 360), angle=(U(r, 10, 30) if overcast else U(r, 0.5, 3)),
                    color=hsv(U(r, 0.07, 0.12), U(r, 0.0, 0.25), 1.0))
    P['overcast'] = overcast
    P['exposure'] = U(r, -0.3, 0.6)
    P['look'] = r.choice(['AgX - Base Contrast', 'AgX - Medium High Contrast', 'AgX - High Contrast', 'AgX - Punchy'])
    for k, v in vals.items():
        set_val(k, v)
    P['colors'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vals.items()}
    P['sunny01'] = 0.15 if overcast else clamp(P['sun']['energy'] / 5.0) * math.sin(math.radians(P['sun']['elev'])) ** 0.5
    # trava - geometrija
    lean_a, lean_d = U(r, 0, 0.45), U(r, 0, 2 * math.pi)
    for k, v in dict(HEIGHT=P['grass_h'], HEIGHT_VAR=U(r, 0.15, 0.5), MOW_MIN=U(r, 0.4, 1.0), N2_SCALE=U(r, 0.1, 0.6),
                     TILT=U(r, 0.1, 0.45), LEAN_X=lean_a * math.cos(lean_d), LEAN_Y=lean_a * math.sin(lean_d),
                     SEED=r.randint(0, 100000), FAR=0.0, NEAR_D=TUFT_NEAR, FAR_D=TUFT_FAR, REED_H=U(r, 0.9, 1.8),
                     HERB_SCALE=U(r, 0.7, 1.3), DENS_R=U(r, 20, 80), WL=WATER_LEVEL,
                     # teksturna trava (vecina podlage)
                     LEAN_DIR=lean_d, GT_STRETCH=U(r, 0.18, 0.5), GT_FINE=U(r, 60, 180),
                     GT_CONTRAST=U(r, 0.2, 0.55), GT_CLUMP=U(r, 2.0, 9.0), GT_CLUMP_AMT=U(r, 0.0, 0.35),
                     GT_BUMP=U(r, 0.2, 0.8), GT_STRAW=U(r, 0.0, 0.5) * (0.4 + season),
                     FL_SCALE=U(r, 15, 45), FL_DENS=(U(r, 0.0, 0.25) if r.random() < 0.6 else 0.0),
                     FL_SIZE=U(r, 0.05, 0.12), FL_MIX=r.random()).items():
        set_val(k, v); P[k] = v
    return P


def thermal_params(r, P):
    s = P['sunny01']
    T_air = U(r, 8, 32)
    T = dict(air=T_air)
    T['veg'] = T_air + U(r, -3, 3)
    T['soil'] = T['veg'] + U(r, 0, 12) * s + U(r, -1, 1)
    T['mud'] = T_air - U(r, 0, 4)
    T['water'] = T_air - U(r, 1, 7)
    T['tree'] = T_air - U(r, 0, 5)
    T['stone'] = T['veg'] + U(r, -2, 12) * s
    T['mole'] = T['soil'] + U(r, -2, 2)
    T['dT_turtle'] = U(r, 1.5, 7.0)        # zelva je toplejsa od okolice (variacija)
    T['turtle'] = T['veg'] + T['dT_turtle']
    T['turtleskin'] = T['turtle'] - U(r, 0, 1.5)
    lo = min(T['veg'], T['water'], T['tree'], T['mud']) - U(r, 0.5, 4)
    hi = max(T['turtle'], T['soil'], T['stone']) + U(r, 0.5, 4)
    g = lambda t: 0.05 + 0.75 * clamp((t - lo) / (hi - lo))   # pusti rezervo, da se render ne odreze pri 1.0
    span = (hi - lo) / 0.75
    for k, var, rv in [('veg', U(r, 0.3, 1.0), U(r, 0.2, 0.7)), ('soil', U(r, 0.5, 2), 0), ('mud', 0.5, 0),
                       ('water', U(r, 0.2, 0.8), 0), ('tree', U(r, 0.5, 1.5), 0.8), ('stone', 0.5, U(r, 1, 4)),
                       ('mole', 1.0, 1.0), ('turtle', U(r, 0.1, 0.6), 0), ('turtleskin', 0.3, 0)]:
        set_val(f'TH_{k}_B', g(T[k])); set_val(f'TH_{k}_V', var / span); set_val(f'TH_{k}_R', rv / span)
    set_val('TH_sky_B', 0.0)
    T['range'] = [lo, hi]
    T['solar'] = U(r, 0.0, 0.08) * s
    return T


# ---------------------------------------------------------------- camera
def cam_dir_for_pixel(cam, scene, u, v):
    fr = cam.data.view_frame(scene=scene)  # TR, BR, BL, TL
    tr, bl = fr[0], fr[2]
    d = Vector((bl.x + (tr.x - bl.x) * u, bl.y + (tr.y - bl.y) * v, bl.z))
    return (cam.matrix_world.to_3x3() @ d).normalized()


def footprint(cam, scene, maxd=70.0):
    fr = cam.data.view_frame(scene=scene)
    Mw = cam.matrix_world; o = Mw.translation; R3 = Mw.to_3x3()
    pts = []
    for c in list(fr) + [(fr[0] + fr[2]) / 2]:
        d = (R3 @ c).normalized()
        t = min(maxd, (o.z / -d.z) if d.z < -1e-3 else maxd)
        p = o + d * t; pts.append(Vector((p.x, p.y)))
    cen = sum(pts, Vector((0, 0))) / len(pts)
    return cen, max((p - cen).length for p in pts) + 2.0


def place_camera_fixed(scene, cam):
    """Kamera je vedno na istem mestu (CAM_LOCATION, CAM_TILT_DEG, CAM_YAW_DEG, CAM_FOCAL)."""
    scene.render.resolution_x, scene.render.resolution_y = RGB_RES
    cam.data.lens = CAM_FOCAL
    R = Matrix.Rotation(math.radians(CAM_YAW_DEG), 3, 'Z') @ Matrix.Rotation(math.radians(CAM_TILT_DEG), 3, 'X')
    cam.matrix_world = Matrix.Translation(CAM_LOCATION) @ R.to_4x4()
    bpy.context.view_layer.update()
    return dict(height=CAM_LOCATION[2] - terrain_height(CAM_LOCATION[0], CAM_LOCATION[1]), tilt_deg=CAM_TILT_DEG,
                yaw_deg=CAM_YAW_DEG, roll_deg=0.0, focal_mm=CAM_FOCAL)


def ground_hit(cam, scene, u, v, maxd=200.0):
    """Presecisce zarka skozi piksel (u, v) [0..1, v navzgor] s terenom; None, ce zadene nebo."""
    o = cam.matrix_world.translation.copy()
    d = cam_dir_for_pixel(cam, scene, u, v)
    if d.z > -1e-3:
        return None
    t = (o.z - 0.0) / -d.z
    for _ in range(8):  # iteracija na valovit teren
        p = o + d * t
        t = (o.z - terrain_height(p.x, p.y)) / -d.z
        if t > maxd:
            return None
    return o + d * t


def sample_ground_point(r, cam, scene, max_dist, margin=0.04, cond=None, tries=400):
    """Nakljucna tocka na tleh, ki je v kadru (enakomerno po sliki), ne v vodi in ne predalec."""
    co = cam.matrix_world.translation
    for _ in range(tries):
        u, v = U(r, margin, 1 - margin), U(r, margin, 1 - margin)
        p = ground_hit(cam, scene, u, v)
        if p is None or (p.xy - co.xy).length > max_dist or in_pond(p.x, p.y):
            continue
        if abs(p.x) > MEADOW_X / 2 - 1 or abs(p.y) > MEADOW_Y / 2 - 1:
            continue
        if cond and not cond(p):
            continue
        return p
    return None


# ---------------------------------------------------------------- sample
def in_pond(x, y):
    return terrain_height(x, y) < WATER_LEVEL + 0.06


def pick_turtle_pos(r, cam, scene):
    """Nakljucno mesto zelve v kadru fiksne kamere (enakomerno po sliki)."""
    p = None
    if r.random() < NEAR_POND_PROB:
        near = lambda q: ((q.x - POND['cx']) / POND['rx']) ** 2 + ((q.y - POND['cy']) / POND['ry']) ** 2 < 2.2
        p = sample_ground_point(r, cam, scene, MAX_TURTLE_DIST, cond=near)
    if p is None:
        p = sample_ground_point(r, cam, scene, MAX_TURTLE_DIST, tries=2000)
    if p is None or in_pond(p.x + 0.2, p.y) or in_pond(p.x - 0.2, p.y):
        p = sample_ground_point(r, cam, scene, MAX_TURTLE_DIST, tries=2000,
                                cond=lambda q: not in_pond(q.x + 0.2, q.y) and not in_pond(q.x - 0.2, q.y))
    return p.x, p.y


def place_turtle(r, root, parts, cam, scene):
    x, y = pick_turtle_pos(r, cam, scene)
    L = U(r, *TURTLE_LEN); s = L / 0.2
    yaw = U(r, 0, 2 * math.pi)
    n = terrain_normal(x, y)
    q = Vector((0, 0, 1)).rotation_difference(n) @ Quaternion((0, 0, 1), yaw)
    q = q @ Quaternion((1, 0, 0), U(r, -0.08, 0.08)) @ Quaternion((0, 1, 0), U(r, -0.08, 0.08))
    z = terrain_height(x, y) - U(r, 0.0, 0.012) * s
    root.matrix_world = Matrix.Translation((x, y, z)) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((s, s, s, 1))
    pose_turtle(parts, r)
    return dict(x=x, y=y, z=z, yaw_deg=math.degrees(yaw) % 360, length_m=L, scale=s)


def place_props(r, props, cam, scene, turtle_xy):
    by_kind = {}
    for ob in props:
        ob.hide_render = True; ob.hide_viewport = True; ob.location = (0, 0, -50)
        by_kind.setdefault(ob['kind'], []).append(ob)
    counts = dict(stone=r.randint(0, 15), mole=r.randint(0, 6) if r.random() < 0.6 else 0, stick=r.randint(0, 6))
    placed = []
    for kind, n in counts.items():
        for ob in r.sample(by_kind[kind], min(n, len(by_kind[kind]))):
            for _ in range(3):
                p = sample_ground_point(r, cam, scene, MAX_TURTLE_DIST * 1.5, margin=0.0, tries=50,
                                        cond=lambda q: (q.xy - Vector(turtle_xy)).length > 0.6)
                if p is None:
                    continue
                x, y = p.x, p.y
                z = terrain_height(x, y)
                if kind == 'mole':  z -= ob.scale.z * 0.35
                elif kind == 'stone': z -= ob.scale.z * U(r, 0.2, 0.6)
                else: z += 0.01
                ob.location = (x, y, z); ob.rotation_euler = (0, 0, U(r, 0, 6.28))
                ob.hide_render = False; ob.hide_viewport = False
                placed.append(dict(kind=kind, x=x, y=y)); break
    return placed


def set_mode(scene, mode, P, T):
    sun = bpy.data.objects['TD_Sun']
    set_val('MODE_T', 1.0 if mode == 'thermal' else 0.0)
    set_val('MODE_M', 1.0 if mode == 'mask' else 0.0)
    if mode == 'rgb':
        sun.data.energy = P['sun']['energy']; sun.data.color = P['sun']['color']
        set_view(scene, ['AgX', 'Filmic', 'Standard'], P['exposure'], P['look'])
        scene.render.resolution_x, scene.render.resolution_y = RGB_RES
        set_samples(scene, SAMPLES_RGB)
        scene.render.image_settings.color_mode = 'RGB'; scene.render.image_settings.color_depth = '8'
    elif mode == 'thermal':
        sun.data.energy = 1.0; sun.data.color = (1, 1, 1)
        set_val('SOLAR', T['solar'])
        set_view(scene, ['Raw', 'Standard'], 0.0)
        scene.render.resolution_x, scene.render.resolution_y = THERMAL_RES
        set_samples(scene, SAMPLES_THERMAL)
        scene.render.image_settings.color_mode = 'RGB'; scene.render.image_settings.color_depth = '16'  # B = maska
    else:
        sun.data.energy = 0.0
        set_view(scene, ['Raw', 'Standard'], 0.0)
        scene.render.resolution_x, scene.render.resolution_y = RGB_RES
        set_samples(scene, SAMPLES_MASK)
        scene.render.image_settings.color_mode = 'BW'; scene.render.image_settings.color_depth = '8'


def render_to(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def load_gray(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = 'Non-Color'
    except TypeError:
        pass
    w, h = img.size
    a = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(h, w, img.channels)[::-1, :, 0].copy()
    bpy.data.images.remove(img)
    return a


def load_rgb(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = 'Non-Color'   # surove vrednosti iz datoteke
    except TypeError:
        pass
    w, h = img.size
    a = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(h, w, img.channels)[::-1, :, :3]
    bpy.data.images.remove(img)
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def load_float(path):
    """Surove float vrednosti (h, w, 3), vrstice od zgoraj."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = 'Non-Color'
    except TypeError:
        pass
    w, h = img.size
    a = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(h, w, img.channels)[::-1, :, :3].copy()
    bpy.data.images.remove(img)
    return a


def downsample(a, out_w, out_h):
    """Povprecenje blokov (npr. 1280x1024 -> 640x512)."""
    h, w = a.shape[:2]
    fy, fx = h // out_h, w // out_w
    if fy * out_h == h and fx * out_w == w:
        return a.reshape(out_h, fy, out_w, fx, *a.shape[2:]).mean(axis=(1, 3))
    yi = (np.arange(out_h) * h / out_h).astype(int); xi = (np.arange(out_w) * w / out_w).astype(int)
    return a[yi][:, xi]


def write_png_rgb(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    h, w, _ = arr.shape
    raw = np.hstack([np.zeros((h, 1), np.uint8), arr.reshape(h, w * 3)]).tobytes()
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(raw, 1)) + chunk(b'IEND', b''))


def save_combined(path, rgb_path, thermal):
    rgb = rgb_path if isinstance(rgb_path, np.ndarray) else None
    """RGB levo, termalna desno (povecana na visino RGB), v eni sliki."""
    if rgb is None:
        rgb = load_rgb(rgb_path)
    H, W = rgb.shape[:2]
    th, tw = thermal.shape
    Wt = int(round(tw * H / th))
    yi = np.minimum((np.arange(H) * th / H).astype(int), th - 1)
    xi = np.minimum((np.arange(Wt) * tw / Wt).astype(int), tw - 1)
    t = thermal[yi][:, xi]
    write_png_rgb(path, np.hstack([rgb, np.repeat(t[:, :, None], 3, axis=2)]))


def write_png_gray(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    h, w = arr.shape
    raw = np.hstack([np.zeros((h, 1), np.uint8), arr]).tobytes()
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(raw, 1)) + chunk(b'IEND', b''))


def blur(a, sigma):
    if sigma <= 0.05:
        return a
    rad = max(1, int(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2); k /= k.sum()
    p = np.pad(a, rad, mode='edge')
    tmp = sum(k[i] * p[rad:rad + a.shape[0], i:i + a.shape[1]] for i in range(2 * rad + 1))
    p = np.pad(tmp, rad, mode='edge')
    return sum(k[i] * p[i:i + a.shape[0], rad:rad + a.shape[1]] for i in range(2 * rad + 1))


def make_thermal_state(a, rng):
    """Parametri termo kamere (AGC, zameglitev, FPN ...). Pri videu so fiksni za ves posnetek."""
    h, w = a.shape
    st = dict(agc=None)
    if rng.random() < 0.7:   # AGC kot pri pravi termo kameri (raztegni histogram)
        lo, hi = np.percentile(a, rng.uniform(0.0, 2.0)), np.percentile(a, rng.uniform(99.7, 99.99))
        hi += (hi - lo) * rng.uniform(0.15, 0.5)   # rezerva, da vroci objekti niso odrezani
        if hi - lo < 0.15:   # ne raztegni enolicne scene v sam sum
            c = (hi + lo) / 2; lo, hi = c - 0.075, c + 0.075
        st['agc'] = (lo, hi)
    st['sigma'] = rng.uniform(0.5, 1.4)
    st['vig'] = rng.uniform(0, 0.08)
    st['gx'], st['gy'] = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
    st['noise'] = rng.uniform(0.004, 0.02)
    st['fpn'] = rng.normal(0, rng.uniform(0.0, 0.006), (1, w))
    st['gain'], st['off'] = rng.uniform(0.9, 1.1), rng.uniform(-0.04, 0.04)
    return st


def apply_thermal_state(a, rng, st):
    if st['agc']:
        lo, hi = st['agc']
        a = (a - lo) / (hi - lo)
    a = blur(a, st['sigma'])
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    a = a * (1 - st['vig'] * rr)                                         # vinjetiranje
    a = a + st['gx'] * (xx / w - 0.5) + st['gy'] * (yy / h - 0.5)        # NUC drift
    a = a + rng.normal(0, st['noise'], a.shape) + st['fpn']              # casovni sum + stolpcni sum (FPN)
    a = np.clip((a - 0.5) * st['gain'] + 0.5 + st['off'], 0, 1)
    return (a * 255 + 0.5).astype(np.uint8)


def thermal_post(a, rng):
    return apply_thermal_state(a, rng, make_thermal_state(a, rng))


# ---------------------------------------------------------------- ANIMACIJA: zelva pride iz mlake
ANIM_FPS = 10                   # termo kamere pogosto snemajo ~9 Hz
ANIM_STOP_DIST = (3.5, 6.5)     # m, kako dalec od kamere se zelva ustavi
ANIM_SPEED = (0.08, 0.18)       # m/s osnovna hitrost hoje (vsak posnetek drugacna)
ANIM_SWIM_SPEED = (0.08, 0.2)   # m/s plavanje do obale
ANIM_PAUSE_RATE = 0.02          # verjetnost postanka na sekundo hoje
ANIM_PAUSE = (1.0, 6.0)         # s, dolzina postanka
ANIM_END_HOLD = (2.0, 4.0)      # s, mirovanje na koncu
ANIM_MAX_SECONDS = 900


def th_gray_of(T, t):
    lo, hi = T['range']
    return 0.05 + 0.75 * clamp((t - lo) / (hi - lo))


def find_shore(angle):
    c = Vector((POND['cx'], POND['cy']))
    d = Vector((math.cos(angle) * POND['rx'], math.sin(angle) * POND['ry']))
    k = 0.1
    while k < 2.0:
        p = c + d * k
        if terrain_height(p.x, p.y) > WATER_LEVEL + 0.03:
            return p, d.normalized()
        k += 0.005
    return None, None


def in_view(cam, scene, p3, margin=0.02):
    q = world_to_camera_view(scene, cam, p3)
    return q.z > 0 and margin < q.x < 1 - margin and margin < q.y < 1 - margin


def plan_walk(r, cam, scene, stop_dist=None):
    """Zacetek v mlaki (P0), izhod na obalo (S), cilj (T) blizu kamere na sredini travnika - vse v kadru."""
    co = cam.matrix_world.translation
    to_cam = math.atan2(co.y - POND['cy'], co.x - POND['cx'])
    for _ in range(300):
        S, out_dir = find_shore(to_cam + U(r, -0.7, 0.7))
        if S is None:
            continue
        P0 = S - out_dir * U(r, 0.6, 1.5)          # zacne v vodi, priplava do obale
        if not in_pond(P0.x, P0.y) or not in_view(cam, scene, Vector((P0.x, P0.y, WATER_LEVEL)), 0.03):
            continue
        for _ in range(500):
            hit = ground_hit(cam, scene, U(r, 0.2, 0.8), U(r, 0.05, 0.35))
            if hit is None:
                continue
            d = (hit.xy - co.xy).length
            lo, hi = (stop_dist - 0.4, stop_dist + 0.4) if stop_dist else ANIM_STOP_DIST
            if lo <= d <= hi and not in_pond(hit.x, hit.y):
                return P0, S, hit.xy
    raise RuntimeError('Ni bilo mogoce najti poti zelve od mlake do kamere')


def simulate_walk(r, P0, S, T, fps, L):
    """Semi-nakljucna hoja: plavanje do obale, nato hoja proti cilju z 'blodenjem', postanki, spreminjanjem hitrosti."""
    s = L / 0.2
    dt = 1.0 / fps
    v0, vs = U(r, *ANIM_SPEED), U(r, *ANIM_SWIM_SPEED)
    wander_amp, wander_f, nseed = U(r, 0.3, 1.0), U(r, 0.05, 0.25), U(r, 0, 1000)
    turn_rate = U(r, 0.5, 1.2)
    p = P0.copy()
    first = S if (S - P0).length > 0.01 else T
    heading = math.atan2(first.y - P0.y, first.x - P0.x)
    t, phase, pause, hold, land_t = 0.0, 0.0, 0.0, None, None
    frames = []
    while t < ANIM_MAX_SECONDS:
        h = terrain_height(p.x, p.y)
        in_water = h < WATER_LEVEL - 0.03 * s
        dist_T = (T - p).length
        v = 0.0
        if hold is not None:
            state = 'stop'; hold -= dt
            if hold <= 0:
                break
        elif not in_water and dist_T < 0.08:
            state = 'stop'; hold = U(r, *ANIM_END_HOLD)
        elif pause > 0:
            state = 'pause'; pause -= dt
        else:
            state = 'swim' if in_water else 'walk'
            if not in_water and r.random() < ANIM_PAUSE_RATE * dt:
                pause = U(r, *ANIM_PAUSE)
            target = S if in_water else T
            want = math.atan2(target.y - p.y, target.x - p.x)
            if not in_water:
                want += wander_amp * noise.noise(Vector((t * wander_f, nseed, 0.0))) * min(1.0, dist_T / 1.5)
            dh = (want - heading + math.pi) % (2 * math.pi) - math.pi
            heading += max(-turn_rate * dt, min(turn_rate * dt, dh))
            if in_water:
                v = vs
            else:
                v = v0 * (0.75 + 0.4 * noise.noise(Vector((t * 0.3, nseed + 5.0, 0.0))))
                ahead = terrain_height(p.x + 0.1 * math.cos(heading), p.y + 0.1 * math.sin(heading))
                v *= clamp(1.0 - 1.2 * max((ahead - h) / 0.1, 0.0), 0.55, 1.0)   # v breg pocasneje
                v = min(v, dist_T * 0.6 + 0.01)
            p = p + Vector((math.cos(heading), math.sin(heading))) * v * dt
            phase += v * dt / (0.07 * s) * 2 * math.pi
        if not in_water and land_t is None:
            land_t = t
        z = max(terrain_height(p.x, p.y), WATER_LEVEL - 0.045 * s)   # v vodi plava tik pod gladino
        frames.append(dict(t=t, x=p.x, y=p.y, z=z, heading=heading, phase=phase, speed=v,
                           state=state, in_water=in_water, land_t=land_t))
        t += dt
    return frames


def pose_frame(fr, root, parts, s):
    n = Vector((0, 0, 1)) if fr['in_water'] else terrain_normal(fr['x'], fr['y'])
    q = Vector((0, 0, 1)).rotation_difference(n) @ Quaternion((0, 0, 1), fr['heading'])
    moving = fr['speed'] > 0
    bob = 0.002 * s * abs(math.sin(fr['phase'])) if moving and not fr['in_water'] else 0.0
    root.matrix_world = Matrix.Translation((fr['x'], fr['y'], fr['z'] + bob)) @ q.to_matrix().to_4x4() \
        @ Matrix.Diagonal((s, s, s, 1))
    ph = fr['phase']
    parts['head'].location = (0.12, 0, 0.022)
    parts['head'].rotation_euler = (0, 0, 0.15 * math.sin(ph * 0.5) if moving else 0.0)
    parts['head'].scale = (1, 1, 1)
    amp = (0.6 if fr['in_water'] else 0.45) if moving else 0.0
    for nm, off in (('leg_fl', 0.0), ('leg_br', 0.0), ('leg_fr', math.pi), ('leg_bl', math.pi)):
        x, y = parts[nm]['base']
        parts[nm].location = (x * 1.15, y * 1.2, 0.01)
        parts[nm].rotation_euler = (0, 0, math.copysign(0.5, x * y) + amp * math.sin(ph + off))
        parts[nm].scale = (1, 1, 1)
    parts['tail'].location = (-0.11, 0, 0.012)


def encode_video(frame_dir, out_path, fps):
    """PNG sekvenca -> mp4 (H.264) prek Blenderjevega Video Sequence Editorja (ffmpeg ni potreben)."""
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
    first = os.path.join(frame_dir, files[0])
    img = bpy.data.images.load(first); w, h = img.size; bpy.data.images.remove(img)
    sc = bpy.data.scenes.new('TD_Encode')
    se = sc.sequence_editor_create()
    st = se.strips.new_image('frames', first, 1, 1)
    for f in files[1:]:
        st.elements.append(f)
    sc.frame_start, sc.frame_end = 1, len(files)
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.fps, sc.render.fps_base = fps, 1.0
    set_view(sc, ['Standard'], 0.0)
    ims = sc.render.image_settings
    try:
        ims.media_type = 'VIDEO'
    except Exception:
        pass
    ims.file_format = 'FFMPEG'
    sc.render.ffmpeg.format = 'MPEG4'
    sc.render.ffmpeg.codec = 'H264'
    sc.render.ffmpeg.constant_rate_factor = 'HIGH'
    sc.render.use_sequencer = True
    sc.render.filepath = out_path
    with bpy.context.temp_override(scene=sc):
        bpy.ops.render.render(animation=True, scene=sc.name)
    bpy.data.scenes.remove(sc)


def _baked_inst_group(ob):
    """Lahek GN: instanca predloge na fiksne tocke (rot/scl iz atributov) - brez raztrosa."""
    ng = bpy.data.node_groups.new('TD_BakedInst_' + ob.name, 'GeometryNodeTree')
    ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    B = NT(ng)
    gi, go = B.node('NodeGroupInput'), B.node('NodeGroupOutput')
    oi = B.node('GeometryNodeObjectInfo', transform_space='ORIGINAL')
    _sock(oi.inputs, 'Object').default_value = ob
    _sock(oi.inputs, 'As Instance').default_value = True
    def attr(name):
        n = B.node('GeometryNodeInputNamedAttribute', data_type='FLOAT_VECTOR')
        _sock(n.inputs, 'Name').default_value = name
        return _sock(n.outputs, 'Attribute')
    e2r = B.node('FunctionNodeEulerToRotation'); B.set(e2r.inputs[0], attr('rot'))
    io = B.node('GeometryNodeInstanceOnPoints')
    B.set(_sock(io.inputs, 'Points'), gi.outputs[0])
    B.set(_sock(io.inputs, 'Instance'), _sock(oi.outputs, 'Geometry'))
    B.set(_sock(io.inputs, 'Rotation'), e2r.outputs[0])
    B.set(_sock(io.inputs, 'Scale'), attr('scl'))
    B.L.new(io.outputs[0], go.inputs[0])
    return ng


def bake_vegetation(scene):
    """Zamrzne travo/sopi/cvetlice/trst iz TD_Vegetation v fiksne tocke.
    Pri animaciji se rastlinje ne spreminja, zato ga ni treba ob vsakem renderju znova raztrositi
    (to je bilo ~60 % casa renderja majhnega izseka)."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    hosts = [o for o in scene.objects if any(m.type == 'NODES' and m.node_group and m.node_group.name == 'TD_Vegetation'
                                             for m in o.modifiers)]
    host_names = {o.name for o in hosts}
    per = {}
    for inst in dg.object_instances:
        if not inst.is_instance or inst.parent is None or inst.parent.original.name not in host_names:
            continue
        per.setdefault(inst.instance_object.original.name, []).append((inst.matrix_world.copy(), inst.random_id))
    coll = bpy.data.collections.get('TD_BakedVeg')
    if coll is None:
        coll = bpy.data.collections.new('TD_BakedVeg'); scene.collection.children.link(coll)
    for ob in list(coll.objects):
        bpy.data.objects.remove(ob)
    total = 0
    for name, items in per.items():
        n = len(items); total += n
        co, rot, scl = [], [], []
        for m, _ in items:
            loc, q, s = m.decompose()
            co += loc[:]; rot += q.to_euler()[:]; scl += s[:]
        me = bpy.data.meshes.new('TD_Baked_' + name)
        me.vertices.add(n)
        me.vertices.foreach_set('co', co)
        me.attributes.new('rot', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', rot)
        me.attributes.new('scl', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', scl)
        me.attributes.new('id', 'INT', 'POINT').data.foreach_set('value', [rid & 0x3fffffff for _, rid in items])
        pob = bpy.data.objects.new('TD_Baked_' + name, me)
        coll.objects.link(pob)
        mod = pob.modifiers.new('inst', 'NODES')
        mod.node_group = bpy.data.node_groups.get('TD_BakedInst_' + name) or _baked_inst_group(bpy.data.objects[name])
    for o in hosts:
        for m in o.modifiers:
            if m.type == 'NODES' and m.node_group and m.node_group.name == 'TD_Vegetation':
                m.show_render = m.show_viewport = False
    bpy.context.view_layer.update()
    return total


REGION_PAD = 14        # px (pri RGB locljivosti) okoli zelve in sence
REGION_SEAM = 3        # px na robu izseka, ki se zavrzejo (anti-aliasing filter na robu regije)


def turtle_region(scene, cam, parts, sun, pad=REGION_PAD):
    """Normaliziran okvir (u0, v0, u1, v1; v od zgoraj) okoli zelve IN njene sence na tleh."""
    pts = [p.matrix_world @ Vector(c) for p in parts.values() for c in p.bound_box]
    down = (sun.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()   # smer sonce -> tla
    if down.z < -0.05:
        for w in list(pts):
            hgt = max(w.z - terrain_height(w.x, w.y), 0.0) + 0.02
            pts.append(w + down * (hgt / -down.z))
    W, H = RGB_RES
    us, vs = [], []
    for q in pts:
        c = world_to_camera_view(scene, cam, q)
        us.append(c.x); vs.append(1.0 - c.y)
    u0, u1 = clamp(min(us) - pad / W), clamp(max(us) + pad / W)
    v0, v1 = clamp(min(vs) - pad / H), clamp(max(vs) + pad / H)
    if u1 <= u0 or v1 <= v0:
        return None
    return (u0, v0, u1, v1)


def region_px(box, w, h):
    u0, v0, u1, v1 = box
    return (max(0, int(math.floor(u0 * w))), max(0, int(math.floor(v0 * h))),
            min(w, int(math.ceil(u1 * w))), min(h, int(math.ceil(v1 * h))))


def render_region(scene, path, box):
    """Zrenderira in zapise SAMO izsek slike (Render Region + crop). Vrne piksle izseka (x0, y0, x1, y1)."""
    w, h = scene.render.resolution_x, scene.render.resolution_y
    x0, y0, x1, y1 = region_px(box, w, h)
    rd = scene.render
    rd.use_border, rd.use_crop_to_border = True, True
    rd.border_min_x, rd.border_max_x = x0 / w, x1 / w
    rd.border_min_y, rd.border_max_y = 1.0 - y1 / h, 1.0 - y0 / h
    try:
        render_to(scene, path)
    finally:
        rd.use_border = rd.use_crop_to_border = False
    return (x0, y0, x1, y1)


def paste_region(bg, reg, px):
    """Prilepi izsek (velikosti px) na ozadje; robnih REGION_SEAM px ne vzame (razen ob robu slike)."""
    x0, y0, x1, y1 = px
    h, w = bg.shape[:2]
    rh, rw = reg.shape[:2]
    x1, y1 = x0 + min(rw, x1 - x0), y0 + min(rh, y1 - y0)
    s_ = REGION_SEAM
    a0, b0 = (s_ if x0 > 0 else 0), (s_ if y0 > 0 else 0)
    a1, b1 = (x1 - x0 - s_ if x1 < w else x1 - x0), (y1 - y0 - s_ if y1 < h else y1 - y0)
    out = bg.copy()
    if a1 > a0 and b1 > b0:
        out[y0 + b0:y0 + b1, x0 + a0:x0 + a1] = reg[b0:b1, a0:a1]
    return out


BLEND_THR_RGB = 5.0 / 255     # razlika, ki se steje za "spremenjeno" (senca, zelva) - sum je manjsi
BLEND_THR_TH = 0.006
BLEND_GROW = 2                 # px razsiritve maske
BLEND_FEATHER = 2              # px mehkega prehoda


def _dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx or dy:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def blend_region(bg, reg, px, thr, turtle=None):
    """Na ozadje prenese SAMO piksle, ki se v izseku res spremenijo (zelva, njena senca),
    z mehkim robom. Drugje ostane ozadje -> ni vidnega pravokotnika zaradi drugacnega suma.
    Vrne (nova slika, maska zelve v koordinatah izseka ali None)."""
    x0, y0, x1, y1 = px
    h, w = reg.shape[:2]
    x1, y1 = x0 + min(w, x1 - x0), y0 + min(h, y1 - y0)
    sub_bg = bg[y0:y1, x0:x1].astype(np.float32)
    sub = reg[:y1 - y0, :x1 - x0].astype(np.float32)
    scale = 255.0 if bg.dtype == np.uint8 else 1.0
    d = np.abs(sub - sub_bg) / scale
    d = d.max(axis=2) if d.ndim == 3 else d
    d = blur(d, 1.5)   # posamezne iskrice suma se zgladijo, senca (vecja ploskev) ostane
    m = d > thr
    if turtle is not None:
        m |= turtle[:y1 - y0, :x1 - x0]
    pad = BLEND_GROW + BLEND_FEATHER + 1
    mp = np.pad(m, pad)
    mp = _dilate(mp, BLEND_GROW)
    a = blur(mp.astype(np.float32), BLEND_FEATHER * 0.6)[pad:-pad, pad:-pad]
    a = np.clip(a * 1.5, 0, 1)
    # skrajni robovi izseka nikoli (tam je anti-aliasing filter izseka nezanesljiv)
    s_ = REGION_SEAM
    if x0 > 0: a[:, :s_] = 0
    if y0 > 0: a[:s_, :] = 0
    if x1 < bg.shape[1]: a[:, -s_:] = 0
    if y1 < bg.shape[0]: a[-s_:, :] = 0
    if sub.ndim == 3:
        a = a[:, :, None]
    out = bg.copy()
    mixed = sub_bg * (1 - a) + sub * a
    out[y0:y1, x0:x1] = (np.clip(mixed + 0.5, 0, 255).astype(np.uint8) if bg.dtype == np.uint8 else mixed)
    return out


def turtle_pose_key(root, parts):
    vals = [round(v, 5) for row in root.matrix_world for v in row]
    for p in parts.values():
        vals += [round(v, 4) for v in p.rotation_euler] + [round(v, 5) for v in p.location]
    return tuple(vals)


def set_turtle_hidden(root, parts, hidden):
    for ob in [root] + list(parts.values()):
        ob.hide_render = hidden


def render_animation(out, seed=0, fps=ANIM_FPS, stop_dist=None, keep_frames=False, use_cycles=False, preview=0):
    """preview=N: simulira celo pot, zrenderira pa samo N enakomerno razporejenih slicic (brez videa)."""
    import shutil
    scene = bpy.data.scenes['TurtleMeadow']
    setup_render(scene, use_cycles)
    _build_cache()
    r = random.Random(seed * 7919 + 17)
    rng = np.random.default_rng(seed * 7919 + 17)
    cam = scene.camera
    root = bpy.data.objects['Turtle']
    parts = {k: bpy.data.objects['Turtle_' + n] for k, n in
             [('shell', 'Shell'), ('head', 'Head'), ('tail', 'Tail'), ('leg_fl', 'leg_fl'), ('leg_fr', 'leg_fr'),
              ('leg_bl', 'leg_bl'), ('leg_br', 'leg_br')]}
    trees = [o for o in bpy.data.objects if o.name.startswith('TD_Tree')]

    # --- ena nakljucna scena za cel posnetek
    P = randomize_look(r, scene)
    for t in trees:
        t.hide_render = r.random() < 0.08
    caminfo = place_camera_fixed(scene, cam)
    L = U(r, *TURTLE_LEN); s = L / 0.2
    pose_turtle(parts, r)   # nakljucna barva/oblika ostane, poza se animira
    P0, S, T = plan_walk(r, cam, scene, stop_dist)
    frames = simulate_walk(r, P0, S, T, fps, L)
    set_grass_params(r, cam, scene, dict(x=T.x, y=T.y, length_m=L))
    set_val('TR', 0.0)   # brez lukenj v travi (zelva se premika)
    placed = place_props(r, list(bpy.data.collections['TD_Props'].objects), cam, scene, (T.x, T.y))
    path = [Vector((f['x'], f['y'])) for f in frames[::5]]
    for ob in bpy.data.collections['TD_Props'].objects:   # nic kamnov na poti zelve
        if not ob.hide_render and min((ob.location.xy - q).length for q in path) < 0.5:
            ob.hide_render = True; ob.hide_viewport = True; ob.location = (0, 0, -50)
    Th = thermal_params(r, P)
    sun = bpy.data.objects['TD_Sun']
    sun.rotation_euler = (math.radians(90 - P['sun']['elev']), 0, math.radians(P['sun']['azim']))
    sun.data.angle = math.radians(P['sun']['angle'])

    # --- keyframi (animacijo si lahko ogledas v shranjenem scene.blend)
    scene.frame_start, scene.frame_end = 0, len(frames) - 1
    scene.render.fps, scene.render.fps_base = fps, 1.0
    for ob in [root] + list(parts.values()):
        ob.animation_data_clear()
    for i, fr in enumerate(frames):
        pose_frame(fr, root, parts, s)
        for k in ('location', 'rotation_euler', 'scale'):
            root.keyframe_insert(k, frame=i)
        for ob in parts.values():
            ob.keyframe_insert('location', frame=i); ob.keyframe_insert('rotation_euler', frame=i)

    adir = os.path.join(os.path.abspath(out), f'anim_{seed:04d}')
    fdir = {k: os.path.join(adir, 'frames', k) for k in ('rgb', 'thermal', 'mask', 'combined', 'labels')}
    for d in list(fdir.values()) + [os.path.join(adir, 'tmp')]:
        os.makedirs(d, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(adir, 'scene.blend'), copy=True)

    # temperatura zelve: iz vode pride hladnejsa in se na soncu segreva
    tau = U(r, 15, 45)
    T_land0 = Th['veg'] + U(r, 0.3, 1.0)
    T_water = Th['water'] + 0.5
    st = None
    W, H = RGB_RES
    rows = []
    tmp = os.path.join(adir, 'tmp')

    # --- 0) rastlinje zamrzni (sicer se ob vsakem renderju znova raztrosi)
    bake_vegetation(scene)

    # --- 1) staticno ozadje BREZ zelve: zrenderira se samo enkrat
    # termalna se renderira v RGB locljivosti (R = temperatura, B = maska) in se nato pomanjsa
    print('Renderiram ozadje (enkrat) ...', flush=True)
    TW, TH = THERMAL_RES
    def thermal_mode():
        set_mode(scene, 'thermal', P, Th)
        scene.render.resolution_x, scene.render.resolution_y = RGB_RES
    set_turtle_hidden(root, parts, True)
    scene.frame_set(0)
    set_mode(scene, 'rgb', P, Th)
    render_to(scene, os.path.join(tmp, 'bg_rgb.png'))
    bg_rgb = load_rgb(os.path.join(tmp, 'bg_rgb.png'))
    thermal_mode()
    render_to(scene, os.path.join(tmp, 'bg_thermal.png'))
    bg_th = load_float(os.path.join(tmp, 'bg_thermal.png'))
    bg_th[:, :, 2] = 0.0
    set_turtle_hidden(root, parts, False)

    # --- 2) vsaka slicica: samo izsek okoli zelve (+ senca), prilepljen na ozadje
    rgb_noise = rng.uniform(1.0, 2.5)   # sum senzorja RGB kamere (v 0..255)
    t0 = time.time()
    prev = dict(key=None, tkey=None)
    n_rendered = 0
    sel = None
    if preview:
        sel = {int(round(k * (len(frames) - 1) / max(preview - 1, 1))) for k in range(preview)}
        print(f'Predogled: {len(sel)} slicic od {len(frames)} ({len(frames) / fps:.0f} s posnetka)', flush=True)
    done = 0
    for i, fr in enumerate(frames):
        if sel is not None and i not in sel:
            continue
        scene.frame_set(i)
        if fr['in_water'] or fr['land_t'] is None:
            temp = T_water
        else:
            temp = Th['turtle'] + (T_land0 - Th['turtle']) * math.exp(-(fr['t'] - fr['land_t']) / tau)
        set_val('TH_turtle_B', th_gray_of(Th, temp)); set_val('TH_turtleskin_B', th_gray_of(Th, temp - 0.5))
        fn = f'f_{i:05d}.png'
        box = turtle_region(scene, cam, parts, sun)
        key = (turtle_pose_key(root, parts), box)
        tkey = (key, round(temp, 1))
        if box is None:   # zelva izven kadra
            comp_rgb, comp_th = bg_rgb, bg_th
        else:
            if tkey != prev['tkey']:        # termalna (+ maska zelve v B kanalu)
                thermal_mode()
                pt = render_region(scene, os.path.join(tmp, 'reg_th.png'), box)
                reg_th = load_float(os.path.join(tmp, 'reg_th.png'))
                prev['tmask'] = reg_th[:, :, 2] > 0.02
                prev['th'] = blend_region(bg_th, reg_th, pt, BLEND_THR_TH, prev['tmask'])
            if key != prev['key']:          # zelva se je premaknila -> RGB izsek
                set_mode(scene, 'rgb', P, Th)
                pr = render_region(scene, os.path.join(tmp, 'reg_rgb.png'), box)
                prev['rgb'] = blend_region(bg_rgb, load_rgb(os.path.join(tmp, 'reg_rgb.png')), pr,
                                           BLEND_THR_RGB, prev['tmask'])
                n_rendered += 1
            prev['key'], prev['tkey'] = key, tkey
            comp_rgb, comp_th = prev['rgb'], prev['th']
        m = comp_th[:, :, 2] > 0.5
        th_small = downsample(comp_th[:, :, 0], TW, TH)
        if keep_frames or preview:
            write_png_gray(os.path.join(fdir['mask'], fn), m.astype(np.uint8) * 255)
        if rgb_noise > 0:
            comp_rgb = np.clip(comp_rgb + rng.normal(0, rgb_noise, comp_rgb.shape[:2])[:, :, None]
                               + rng.normal(0, rgb_noise * 0.4, comp_rgb.shape), 0, 255).astype(np.uint8)
        write_png_rgb(os.path.join(fdir['rgb'], fn), comp_rgb)
        if st is None:   # parametri termo kamere fiksni za cel posnetek -> brez utripanja
            st = make_thermal_state(th_small, rng)
        th = apply_thermal_state(th_small, rng, st)   # casovni sum je na vsaki slicici nov
        write_png_gray(os.path.join(fdir['thermal'], fn), th)
        save_combined(os.path.join(fdir['combined'], fn), comp_rgb, th)
        # oznake
        vis = int(m.sum())
        c2 = world_to_camera_view(scene, cam, root.matrix_world @ Vector((0, 0, 0.03)))
        bb = None
        if vis > 0:
            ys, xs = np.nonzero(m)
            bb = [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H]
        with open(os.path.join(fdir['labels'], fn.replace('.png', '.txt')), 'w') as f:
            if bb:
                f.write(f'0 {(bb[0] + bb[2]) / 2:.6f} {(bb[1] + bb[3]) / 2:.6f} {bb[2] - bb[0]:.6f} {bb[3] - bb[1]:.6f}\n')
        rows.append([i, f"{fr['t']:.3f}", f"{fr['x']:.4f}", f"{fr['y']:.4f}", f"{fr['z']:.4f}",
                     f"{math.degrees(fr['heading']) % 360:.1f}", f"{fr['speed']:.4f}", fr['state'], vis,
                     f'{c2.x:.5f}', f'{1 - c2.y:.5f}', *[('' if bb is None else f'{b:.5f}') for b in (bb or [None] * 4)],
                     f'{temp:.2f}'])
        done += 1
        n_total = len(sel) if sel else len(frames)
        if sel is not None or done % 20 == 0 or done == n_total:
            el = time.time() - t0
            print(f'  slicica {done}/{n_total} (f{i})  {fr["state"]}  {el / done:.2f} s/slicico  '
                  f'ETA {el / done * (n_total - done) / 60:.1f} min', flush=True)
    print(f'Izsekov zrenderiranih: {n_rendered} od {len(frames)} slicic', flush=True)


    with open(os.path.join(adir, 'track.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 'time_s', 'turtle_x', 'turtle_y', 'turtle_z', 'heading_deg', 'speed_mps', 'state',
                     'visible_px', 'center_u', 'center_v', 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1', 'turtle_temp_C'])
        wr.writerows(rows)
    with open(os.path.join(adir, 'info.json'), 'w') as f:
        json.dump(dict(seed=seed, fps=fps, frames=len(frames), duration_s=len(frames) / fps, turtle_length_m=L,
                       start_in_pond=list(P0), shore=list(S), end=list(T), camera=caminfo,
                       rgb_res=list(RGB_RES), thermal_res=list(THERMAL_RES),
                       scene=dict(grass_height_m=P['grass_h'], overcast=P['overcast'], sun=P['sun'], props=placed),
                       thermal_celsius=Th), f, indent=1, default=lambda o: list(o) if hasattr(o, '__iter__') else str(o))
    if preview:
        shutil.rmtree(os.path.join(adir, 'tmp'), ignore_errors=True)
        print('Predogled shranjen (brez videa):', os.path.join(adir, 'frames'))
        return
    print('Kodiram video ...', flush=True)
    for k in ('rgb', 'thermal', 'combined'):
        encode_video(fdir[k], os.path.join(adir, f'{k}.mp4'), fps)
    shutil.rmtree(os.path.join(adir, 'tmp'), ignore_errors=True)
    if not keep_frames:
        shutil.rmtree(os.path.join(adir, 'frames'), ignore_errors=True)
    print('Animacija shranjena:', adir, f'({len(frames)} slicic, {len(frames) / fps:.1f} s)')


def set_grass_params(r, cam, scene, tinfo, dens=None):
    """3D sopi trave samo v vidnem polju, blizu kamere in redko (vecina trave je tekstura)."""
    cen, R = footprint(cam, scene)
    hfov = cam.data.angle_x
    area = 0.5 * hfov * TUFT_FAR ** 2           # priblizna ploscina vidnega klina do TUFT_FAR
    if dens is None:
        dens = U(r, *TUFT_DENSITY) if r.random() > 0.25 else U(r, 0.0, 3.0)
    dens = min(dens, MAX_GRASS_INSTANCES / max(area * 0.5, 1.0))
    for k, v in dict(CX=cen.x, CY=cen.y, R=R, CAMX=cam.location.x, CAMY=cam.location.y,
                     DENS_G=dens, DENS_H=dens * U(r, 0.0, 0.08), TX=tinfo['x'], TY=tinfo['y'],
                     TR=0.42 * tinfo['length_m']).items():
        set_val(k, v)
    return cen, R, dens


def render_sample(idx, seed, out, scene, csv_rows):
    r = random.Random(seed * 1_000_003 + idx)
    rng = np.random.default_rng(seed * 1_000_003 + idx)
    cam = scene.camera
    root = bpy.data.objects['Turtle']
    parts = {k: bpy.data.objects['Turtle_' + n] for k, n in
             [('shell', 'Shell'), ('head', 'Head'), ('tail', 'Tail'), ('leg_fl', 'leg_fl'), ('leg_fr', 'leg_fr'),
              ('leg_bl', 'leg_bl'), ('leg_br', 'leg_br')]}
    trees = [o for o in bpy.data.objects if o.name.startswith('TD_Tree')]
    name = f'img_{idx:05d}'                      # zacasno ime, koncno ime = koordinate zelve
    raw_t = os.path.join(out, 'tmp', name + '_traw.png')
    raw_m = os.path.join(out, 'tmp', name + '_mask.png')
    negative = r.random() < NEGATIVE_PROB

    P = randomize_look(r, scene)
    # drevesa: nekaj nakljucno skritih
    for t in trees:
        t.hide_render = r.random() < 0.08

    caminfo = place_camera_fixed(scene, cam)
    for attempt in range(6):
        tinfo = place_turtle(r, root, parts, cam, scene)
        target = Vector((tinfo['x'], tinfo['y'], tinfo['z'] + 0.03 * tinfo['scale']))
        cen, R, dens = set_grass_params(r, cam, scene, tinfo)
        props = place_props(r, list(bpy.data.collections['TD_Props'].objects), cam, scene, (tinfo['x'], tinfo['y']))
        if negative:
            root.location.z -= 50  # zelva pod zemljo -> slika brez zelve
        bpy.context.view_layer.update()
        T = thermal_params(r, P)
        set_mode(scene, 'mask', P, T)
        render_to(scene, raw_m)
        m = load_gray(raw_m) > 0.5
        vis = int(m.sum())
        if negative or vis >= MIN_VISIBLE_PIXELS:
            break
        print(f'  {name}: zelva skoraj nevidna ({vis} px), nov poskus')

    os.remove(raw_m)
    # ime datoteke = koordinate zelve (svetovni X, Y v metrih) + zaporedna stevilka
    name = f"none_{idx:05d}" if negative else f"x{tinfo['x']:+.3f}_y{tinfo['y']:+.3f}_{idx:05d}"
    paths = {k: os.path.join(out, k, name + '.png') for k in ('rgb', 'thermal', 'mask', 'combined')}
    write_png_gray(paths['mask'], m.astype(np.uint8) * 255)
    # RGB
    set_mode(scene, 'rgb', P, T)
    sun = bpy.data.objects['TD_Sun']
    sun.rotation_euler = (math.radians(90 - P['sun']['elev']), 0, math.radians(P['sun']['azim']))
    sun.data.angle = math.radians(P['sun']['angle'])
    render_to(scene, paths['rgb'])
    # termalna
    set_mode(scene, 'thermal', P, T)
    render_to(scene, raw_t)
    thermal = thermal_post(load_gray(raw_t), rng)
    write_png_gray(paths['thermal'], thermal)
    save_combined(paths['combined'], paths['rgb'], thermal)
    os.remove(raw_t)

    # oznake
    W, H = RGB_RES
    scene.render.resolution_x, scene.render.resolution_y = RGB_RES
    c2 = world_to_camera_view(scene, cam, target)
    proj = []
    for p in parts.values():
        for c in p.bound_box:
            q = world_to_camera_view(scene, cam, p.matrix_world @ Vector(c)); proj.append((q.x, 1 - q.y))
    px = [p[0] for p in proj]; py = [p[1] for p in proj]
    bbox_proj = [clamp(min(px)), clamp(min(py)), clamp(max(px)), clamp(max(py))]
    visible = (not negative) and vis >= 1
    bbox_mask = None
    if visible:
        ys, xs = np.nonzero(m)
        bbox_mask = [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H]
    with open(os.path.join(out, 'labels', name + '.txt'), 'w') as f:
        if visible:
            x0, y0, x1, y1 = bbox_mask
            f.write(f'0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n')
    slant = (target - cam.location).length
    gsd = slant * 36.0 / cam.data.lens / W   # m/px na razdalji zelve
    meta = dict(
        id=name, files={k: os.path.relpath(v, out) for k, v in paths.items()},
        turtle=dict(present=not negative, world=dict(x=tinfo['x'], y=tinfo['y'], z=tinfo['z']),
                    yaw_deg=tinfo['yaw_deg'], length_m=tinfo['length_m'], visible=visible, visible_pixels_rgb=vis,
                    center_norm=[c2.x, 1 - c2.y], center_px_rgb=[c2.x * W, (1 - c2.y) * H],
                    center_px_thermal=[c2.x * THERMAL_RES[0], (1 - c2.y) * THERMAL_RES[1]],
                    bbox_mask_norm=bbox_mask, bbox_projected_norm=bbox_proj),
        camera=dict(location=list(cam.location), rotation_euler_deg=[math.degrees(a) for a in cam.rotation_euler],
                    distance_to_turtle_m=slant, gsd_at_turtle_m_per_px_rgb=gsd, **caminfo),
        image=dict(rgb_res=list(RGB_RES), thermal_res=list(THERMAL_RES)),
        scene=dict(grass_height_m=P['grass_h'], overcast=P['overcast'], sun=P['sun'], exposure=P['exposure'],
                   bare_threshold=P['colors']['BARE_THR'], dry_fraction=P['colors']['G_DRY_FRAC'],
                   grass_patch=dict(center=[cen.x, cen.y], radius=R, density=dens), props=props),
        thermal_celsius={k: v for k, v in T.items()},
        seed=seed,
    )
    with open(os.path.join(out, 'meta', name + '.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=lambda o: list(o) if hasattr(o, '__iter__') else str(o))
    bm = bbox_mask or [None] * 4
    csv_rows.append([name, int(not negative), int(visible), vis, f"{tinfo['x']:.3f}", f"{tinfo['y']:.3f}", f"{tinfo['z']:.3f}",
                     f"{tinfo['yaw_deg']:.1f}", f"{tinfo['length_m']:.3f}", f'{c2.x:.5f}', f'{1 - c2.y:.5f}',
                     *[('' if b is None else f'{b:.5f}') for b in bm],
                     f'{cam.location.x:.2f}', f'{cam.location.y:.2f}', f'{cam.location.z:.2f}',
                     f"{caminfo['height']:.2f}", f"{caminfo['tilt_deg']:.1f}", f"{caminfo['focal_mm']:.1f}",
                     f"{P['grass_h']:.3f}", f"{T['dT_turtle']:.2f}"])


CSV_HEADER = ['id', 'turtle_present', 'visible', 'visible_px', 'turtle_x', 'turtle_y', 'turtle_z', 'yaw_deg', 'length_m',
              'center_u', 'center_v', 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1', 'cam_x', 'cam_y', 'cam_z',
              'cam_height', 'tilt_deg', 'focal_mm', 'grass_h', 'turtle_dT']


def render_dataset(n, out, seed=0, start=0, use_cycles=False):
    scene = bpy.data.scenes['TurtleMeadow']
    setup_render(scene, use_cycles)
    out = os.path.abspath(out)
    for d in ('rgb', 'thermal', 'mask', 'combined', 'labels', 'meta', 'tmp'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with open(os.path.join(out, 'classes.txt'), 'w') as f:
        f.write('turtle\n')
    csv_path = os.path.join(out, 'annotations.csv')
    new = not os.path.exists(csv_path)
    _build_cache()
    t0 = time.time()
    with open(csv_path, 'a', newline='') as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(CSV_HEADER)
        for i in range(start, start + n):
            rows = []
            ts = time.time()
            render_sample(i, seed, out, scene, rows)
            wr.writerows(rows); f.flush()
            el = time.time() - t0; done = i - start + 1
            print(f'[{done}/{n}] img_{i:05d}  {time.time() - ts:.1f}s  (ETA {el / done * (n - done) / 60:.1f} min)', flush=True)
    try:
        os.rmdir(os.path.join(out, 'tmp'))
    except OSError:
        pass


def preview(seed=0):
    """Postavi en nakljucen primer v sceno (brez renderja) - za ogled v Blenderju."""
    scene = bpy.data.scenes['TurtleMeadow']
    _build_cache()
    r = random.Random(seed)
    P = randomize_look(r, scene)
    root = bpy.data.objects['Turtle']
    parts = {k: bpy.data.objects['Turtle_' + n] for k, n in
             [('shell', 'Shell'), ('head', 'Head'), ('tail', 'Tail'), ('leg_fl', 'leg_fl'), ('leg_fr', 'leg_fr'),
              ('leg_bl', 'leg_bl'), ('leg_br', 'leg_br')]}
    trees = [o for o in bpy.data.objects if o.name.startswith('TD_Tree')]
    place_camera_fixed(scene, scene.camera)
    tinfo = place_turtle(r, root, parts, scene.camera, scene)
    set_grass_params(r, scene.camera, scene, tinfo)
    place_props(r, list(bpy.data.collections['TD_Props'].objects), scene.camera, scene, (tinfo['x'], tinfo['y']))
    T = thermal_params(r, P)
    set_mode(scene, 'rgb', P, T)
    sun = bpy.data.objects['TD_Sun']
    sun.rotation_euler = (math.radians(90 - P['sun']['elev']), 0, math.radians(P['sun']['azim']))


# ---------------------------------------------------------------- main
def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    def arg(name, default=None, typ=str):
        if name in argv:
            i = argv.index(name)
            return typ(argv[i + 1]) if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
        return default
    seed = arg('--seed', 0, int)
    if arg('--low'):   # nizka kvaliteta za hiter test: manjsa slika, manj vzorcev
        global RGB_RES, SAMPLES_RGB, SAMPLES_THERMAL, SAMPLES_MASK, REGION_PAD
        RGB_RES = (THERMAL_RES[0], THERMAL_RES[1])
        SAMPLES_RGB, SAMPLES_THERMAL, SAMPLES_MASK = 4, 2, 1
        REGION_PAD = 8
        print('Nizka kvaliteta:', RGB_RES, 'vzorci', SAMPLES_RGB, SAMPLES_THERMAL)
    if arg('--build'):
        build_scene(seed=1)
        preview(seed)
        blend = os.path.abspath(arg('--blend', 'turtle_meadow.blend'))
        bpy.ops.wm.save_as_mainfile(filepath=blend)
        print('Shranjeno:', blend)
    n = arg('--render', None, int)
    if n:
        if 'TurtleMeadow' not in bpy.data.scenes:
            build_scene(seed=1)
        render_dataset(n, arg('--out', 'dataset'), seed=seed, start=arg('--start', 0, int), use_cycles=bool(arg('--cycles')))
    if arg('--animate') or arg('--preview'):
        if 'TurtleMeadow' not in bpy.data.scenes:
            build_scene(seed=1)
        render_animation(arg('--out', 'animations'), seed=seed, fps=arg('--fps', ANIM_FPS, int),
                         stop_dist=arg('--stop-dist', None, float), keep_frames=bool(arg('--keep-frames')),
                         use_cycles=bool(arg('--cycles')), preview=arg('--preview', 0, int))
    if not argv and 'TurtleMeadow' in bpy.data.scenes:
        preview(random.randint(0, 10 ** 6))   # zagon iz Text Editorja: nov nakljucen primer


if __name__ == '__main__':
    main()

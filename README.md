# Simulacije posnetkov sklednic

Sintetični podatki za učenje modela, ki prepozna želvo (močvirsko sklednico) na travniku.
Blender scena + Python skripta generirata **RGB in termalne slike** z oznakami, ter **animacije**,
kjer želva priplava iz mlake in gre čez travnik proti kameri.

![Pot želve od mlake do kamere](primeri/pot_zelve/pot_trajektorija.png)

## Scena

- Travnik 100 × 50 m (**0,5 ha**), rahlo valovit teren, izhodišče (0, 0) je na sredini.
- **Mlaka** ob vzhodnem robu (+X) s trstičjem in rogozom, blato ob obali.
- **Drevesa** (listavci in iglavci) v pasovih ob robovih travnika.
- **Želva**: 3D model, dolžina 15–25 cm, oklep s ščitki in rumenimi pikami, glava/noge iztegnjene ali skrite.
- **Fiksna kamera** na sredini travnika, 5 m visoko, nagib 60°, gleda proti mlaki (24 mm).
- **Trava** je večinoma proceduralna tekstura (bilke, suhe krpe, gola zemlja, cvetlice) + redki 3D šopi blizu kamere.
- Motilci: kamni (na soncu topli – dobri lažni pozitivi), krtine, veje.

Pri vsaki sliki/animaciji se naključno spremenijo: položaj in barva želve, barve trave, zemlje, vode in dreves,
suhost, gole krpe, smer polegnjenosti trave, sonce/oblačnost, rekviziti in temperature v termalni sliki.

## Zahteve

- **Blender 4.2+** (testirano na Blender 5.0.1), EEVEE, GPU.
- Ni drugih odvisnosti (numpy je del Blenderja, video se kodira prek Blenderja – ffmpeg ni potreben).

V ukazih spodaj je `blender` pot do Blenderja, npr. `"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"`.

## 1) Posamezne slike (dataset)

```bash
blender -b turtle_meadow.blend -P turtle_dataset.py -- --render 1000 --out dataset
```

Dodatno: `--seed 7`, `--start 1000` (nadaljuj številčenje), `--cycles`.
Hitrost: ~6 s na sliko (RGB 1280×1024 + termalna 640×512 + maska).

**Ime datoteke = koordinate želve** v metrih: `x+12.135_y-6.025_00003.png` (+ zaporedna številka).

| mapa | vsebina |
|---|---|
| `rgb/` | normalna slika 1280×1024 |
| `thermal/` | termalna slika 640×512, 8-bit sivinska (white-hot) |
| `mask/` | segmentacijska maska vidnega dela želve |
| `combined/` | RGB + termalna ena ob drugi (za pregled) |
| `labels/` | YOLO oznaka `0 cx cy w h` (normalizirano – velja za RGB in termalno) |
| `meta/` | JSON: svetovne koordinate želve, kamera, bbox, vsi parametri in temperature |
| `annotations.csv` | povzetek vseh slik |

Primeri: [`primeri/slike`](primeri/slike).

## 2) Animacija: želva pride iz mlake

```bash
blender -b turtle_meadow.blend -P turtle_dataset.py -- --animate --seed 1 --out animations
```

- Ena naključna scena za cel posnetek. Želva začne v mlaki, priplava na obalo in pol-naključno
  (vijuganje, spreminjanje hitrosti, postanki) odide čez travnik do 3,5–6,5 m pred kamero.
- Realna hitrost ~0,08–0,18 m/s (vsakič drugačna) → posnetek je dolg 3–8 minut.
- V termalni sliki pride iz vode hladnejša in se na soncu postopno segreva.
- Izhod v `animations/anim_0001/`: `rgb.mp4`, `thermal.mp4`, `combined.mp4`,
  `track.csv` (položaj, hitrost, stanje, bbox in temperatura želve za vsako sličico), `info.json`,
  `scene.blend` (animirana scena za ogled v Blenderju).

Možnosti:

| zastavica | pomen |
|---|---|
| `--fps 10` | sličice na sekundo (privzeto 10) |
| `--low` | nizka kvaliteta za hiter test (RGB 640×512, manj vzorcev) |
| `--stop-dist 5` | kako daleč od kamere (m) se želva ustavi |
| `--keep-frames` | obdrži posamezne sličice + YOLO oznake za vsako sličico |
| `--preview 10` | simulira celo pot, zrenderira samo 10 sličic (brez videa) – za hiter pregled poti |

Primer predogleda poti: [`primeri/pot_zelve`](primeri/pot_zelve).

**Hitrost renderja:** ozadje se zrenderira samo enkrat, nato pa za vsako sličico samo majhen izsek
okoli želve in njene sence (Render Region), ki se zmeša na ozadje (samo spremenjeni piksli – brez vidnih robov).
Rastlinje se pred animacijo "zapeče", maska je v B kanalu termalnega renderja.
Približno 0,8 s/sličico (polna kvaliteta) oz. 0,45 s/sličico (`--low`).

## Nastavitve

Na vrhu `turtle_dataset.py`: velikost travnika, mlaka, ločljivosti, položaj in nagib kamere,
velikost želve, gostota 3D šopov, delež slik brez želve (`NEGATIVE_PROB`), parametri animacije (`ANIM_*`) …

Scena se zgradi znova z:

```bash
blender -b -P turtle_dataset.py -- --build --blend turtle_meadow.blend
```

Zagon skripte iz Blenderjevega Text Editorja (v `.blend` je vgrajena) postavi nov naključen primer za ogled.

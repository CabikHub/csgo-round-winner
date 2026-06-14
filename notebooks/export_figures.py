"""
CSGO_FİNAL.ipynb → figures/ HTML Dışa Aktarıcı
================================================
Kullanım:
    python export_figures.py

- Bu script'i veri seti ile aynı klasöre koy:
    klasor/
    ├── data/
    │   └── raw/
    │       └── YBS3259_CSGO_Final_Verisi.csv
    ├── CSGO_FİNAL.ipynb
    └── export_figures.py   ← bu dosya

- Çalıştır:  python export_figures.py
- Çıktılar:  figures/  klasörüne otomatik kaydedilir.
"""

import os
import sys
import json
import re
from pathlib import Path

# ── Gerekli kütüphane kontrolü ──────────────────────────────────────────────
REQUIRED = ["pandas", "numpy", "plotly", "sklearn", "joblib"]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"[HATA] Eksik kütüphane(ler): {', '.join(missing)}")
    print(f"       Kur: pip install {' '.join(missing)}")
    sys.exit(1)

# ── Notebook yolu ────────────────────────────────────────────────────────────
NOTEBOOK = Path("CSGO_FİNAL.ipynb")
if not NOTEBOOK.exists():
    print(f"[HATA] '{NOTEBOOK}' bulunamadı. Script'i notebook ile aynı klasörde çalıştır.")
    sys.exit(1)

# ── figures/ klasörünü oluştur ───────────────────────────────────────────────
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
print(f"[✓] figures/ klasörü hazır: {FIGURES_DIR.resolve()}")

# ── Notebook'u oku ──────────────────────────────────────────────────────────
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
code_cells = [c for c in cells if c["cell_type"] == "code"]
print(f"[✓] Notebook okundu — {len(code_cells)} kod hücresi bulundu.\n")

# ── Yardımcı: Plotly fig.show() → write_html + HTML blokları yakala ─────────
def patch_cell_source(source: str, cell_index: int) -> str:
    """
    1. Her fig.show() satırını otomatik write_html + show çiftiyle değiştirir.
    2. display(HTML(...)) çağrılarının üzerine HTML'i dosyaya da kazan.
    """
    lines = source.splitlines(keepends=True)
    new_lines = []
    fig_counter = [0]   # mutable reference

    for line in lines:
        # fig.show() → write_html + show
        stripped = line.rstrip()
        if re.match(r"^\s*fig\.show\(\)\s*$", stripped):
            indent = len(line) - len(line.lstrip())
            sp = " " * indent
            fig_counter[0] += 1
            fname = f"figures/cell{cell_index:02d}_fig{fig_counter[0]:02d}.html"
            new_lines.append(
                f'{sp}fig.write_html("{fname}", full_html=True, include_plotlyjs="cdn")\n'
            )
            new_lines.append(
                f'{sp}print("[KAYDEDILDI] {fname}")\n'
            )
            new_lines.append(line)   # orijinal show() da çalışsın
        else:
            new_lines.append(line)

    return "".join(new_lines)


# ── HTML display çağrılarını yakala (regex ile) ──────────────────────────────
HTML_PATTERN = re.compile(
    r'display\s*\(\s*HTML\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*\)',
    re.MULTILINE
)

def inject_html_save(source: str, cell_index: int) -> str:
    """
    display(HTML(varname)) çağrılarının öncesine varname'i dosyaya yazan
    satırlar ekler.
    """
    lines = source.splitlines(keepends=True)
    new_lines = []
    counter = [0]

    for line in lines:
        m = HTML_PATTERN.search(line)
        if m:
            varname = m.group(1)
            indent = len(line) - len(line.lstrip())
            sp = " " * indent
            counter[0] += 1
            fname = f"figures/cell{cell_index:02d}_html{counter[0]:02d}.html"
            # Tırnak çakışmasını önlemek için _WRAP_OPEN/_WRAP_CLOSE değişkenleri
            # doğrudan Python kodu olarak enjekte ediliyor (string literal değil)
            new_lines.append(
                f'{sp}_wrap_open = (\n'
                f'{sp}    "<!DOCTYPE html><html><head>"\n'
                f'{sp}    \'<meta charset="utf-8">\'\n'
                f'{sp}    \'<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>\'\n'
                f'{sp}    "</head><body>"\n'
                f'{sp})\n'
                f'{sp}_wrap_close = "</body></html>"\n'
                f'{sp}with open("{fname}", "w", encoding="utf-8") as _f:\n'
                f'{sp}    _f.write(_wrap_open + str({varname}) + _wrap_close)\n'
                f'{sp}print("[KAYDEDILDI] {fname}")\n'
            )
        new_lines.append(line)

    return "".join(new_lines)


# ── Tüm kod hücrelerini sırayla çalıştır ─────────────────────────────────────
namespace = {"__name__": "__main__"}

# IPython stub — gerçek display() çağrılarını sessizce geç (dosyaya kayıt yeter)
from IPython.display import HTML as _HTML  # noqa

class _FakeDisplay:
    def __init__(self, *a, **kw): pass

# ipywidgets stub
class _FakeWidget:
    def __init__(self, *a, **kw): pass
    def observe(self, *a, **kw): pass
    value = 0.5

try:
    import ipywidgets as _iw
except ImportError:
    pass  # zaten kurulu değilse atla

namespace["display"] = lambda *a, **kw: None
namespace["HTML"] = _HTML

total_saved = 0
errors = 0

for idx, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue

    src = "".join(cell["source"]).strip()
    if not src:
        continue

    # ipywidgets satırlarını yoruma al (interaktif widget, script'te çalışmaz)
    src = re.sub(r"(?m)^(\s*)(.*ipywidgets.*)", r"\1# \2  # [ATILDI: widget]", src)
    src = re.sub(r"(?m)^(\s*)(.*\.interact\(.*)", r"\1# \2  # [ATILDI: widget]", src)

    # fig.show() → write_html ile değiştir
    src = patch_cell_source(src, idx)

    # display(HTML(var)) → dosyaya kaydet
    src = inject_html_save(src, idx)

    try:
        exec(compile(src, f"<cell {idx}>", "exec"), namespace)
    except Exception as e:
        print(f"  [⚠]  Hücre {idx} atlandı: {type(e).__name__}: {e}")
        errors += 1
        continue

# ── Özet ─────────────────────────────────────────────────────────────────────
saved_files = sorted(FIGURES_DIR.glob("*.html"))
print("\n" + "="*60)
print(f"  Tamamlandı!")
print(f"  Kaydedilen dosya : {len(saved_files)}")
print(f"  Hücre hatası     : {errors}")
print(f"  Klasör           : {FIGURES_DIR.resolve()}")
print("="*60)
for f in saved_files:
    print(f"  • {f}")

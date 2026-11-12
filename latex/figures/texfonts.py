"""The documents' fonts for matplotlib figures.

latex/wakaru.sty sets headings and labels in IBM Plex Sans, math in newpx
and code in IBM Plex Mono. Figures take the same faces from the TeX
distribution that sets the documents (latex/flake.nix): IBM Plex Sans for
text, TeX Gyre Pagella (the OpenType Palatino that newpx is built on) for
mathtext, and IBM Plex Mono. The files are found with kpsewhich, run
through `nix develop` as the Makefile runs pdflatex, so a figure needs no
font installed on the system.

All three ship as OpenType with CFF outlines, which matplotlib's PDF
backend embeds under `pdf.fonttype: 42` as if they were TrueType: the file
is then malformed (poppler warns of a font type mismatch, and stricter
viewers drop the text). So each is converted once to TrueType outlines; a
face that ships as TrueType would be copied as it is. The files land in
preview/fonts/ beside the figure previews (untracked), and those copies
are what matplotlib loads.

`use()` registers the files and points matplotlib's sans-serif, monospace
and mathtext settings at them; the figure scripts set `font.family` to
sans-serif. Where the distribution cannot be reached it leaves matplotlib's
defaults in place and says so, so a figure script still runs.
"""

import os
import shutil
import subprocess
from pathlib import Path

LATEX = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / "preview" / "fonts"
SANS = "IBM Plex Sans"
MATH = "TeX Gyre Pagella"
MONO = "IBM Plex Mono"
FILES = [
    "IBMPlexSans-Regular.otf", "IBMPlexSans-Italic.otf",
    "IBMPlexSans-SemiBold.otf", "IBMPlexSans-Bold.otf",
    "IBMPlexSans-BoldItalic.otf",
    "texgyrepagella-regular.otf", "texgyrepagella-italic.otf",
    "texgyrepagella-bold.otf", "texgyrepagella-bolditalic.otf",
    "IBMPlexMono-Regular.otf", "IBMPlexMono-Italic.otf",
    "IBMPlexMono-Bold.otf",
]


def find() -> list[str]:
    """Paths of FILES in the flake's TeX distribution; [] if out of reach."""
    cmd = ["kpsewhich", *FILES]
    if "IN_NIX_SHELL" not in os.environ and shutil.which("nix"):
        cmd = ["nix", "develop", str(LATEX), "--command", *cmd]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=900).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [p for p in out.split()
            if p.endswith((".otf", ".ttf")) and os.path.exists(p)]


def to_truetype(src: str, dst: Path) -> None:
    """Rewrite a CFF-flavored OpenType font with TrueType outlines: each
    glyph's cubic curves approximated by quadratics to within one font
    unit, every other table kept."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(src)
    glyphs, order = font.getGlyphSet(), font.getGlyphOrder()
    glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for name in order:
        pen = TTGlyphPen(glyphs)
        # CFF contours wind the other way from TrueType's
        glyphs[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyf[name] = pen.glyph()
    font["loca"], font["glyf"] = newTable("loca"), glyf
    for tag in ("CFF ", "VORG"):
        if tag in font:
            del font[tag]
    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    maxp.maxZones, maxp.maxTwilightPoints, maxp.maxStorage = 1, 0, 0
    maxp.maxFunctionDefs = maxp.maxInstructionDefs = 0
    maxp.maxStackElements = maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = 0
    post = font["post"]
    post.formatType, post.extraNames, post.mapping = 2.0, [], {}
    post.glyphOrder = order
    font.sfntVersion = "\x00\x01\x00\x00"
    dst.parent.mkdir(parents=True, exist_ok=True)
    font.save(dst)


def truetype() -> list[Path]:
    """The cached TrueType faces, copying or converting whichever are
    missing."""
    want = [CACHE / (Path(f).stem + ".ttf") for f in FILES]
    if not all(p.exists() for p in want):
        for src in find():
            dst = CACHE / (Path(src).stem + ".ttf")
            if dst.exists():
                continue
            if src.endswith(".ttf"):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            else:
                to_truetype(src, dst)
    return [p for p in want if p.exists()]


def use() -> bool:
    """Select the documents' fonts for every later figure. Returns whether
    every family was found; call after matplotlib's backend is chosen."""
    import matplotlib
    from matplotlib import font_manager

    for path in truetype():
        font_manager.fontManager.addfont(str(path))
    names = {f.name for f in font_manager.fontManager.ttflist}
    missing = [f for f in (SANS, MATH, MONO) if f not in names]
    if missing:
        print(f"texfonts: {' / '.join(missing)} not found; "
              "figures keep matplotlib's default fonts")
        return False
    default = matplotlib.rcParamsDefault
    matplotlib.rcParams.update({
        # the generic families keep their defaults as fallbacks, for the
        # glyphs (Greek, some symbols) the faces lack
        "font.sans-serif": [SANS] + default["font.sans-serif"],
        "font.serif": [MATH] + default["font.serif"],
        "font.monospace": [MONO] + default["font.monospace"],
        # math is newpx's in the documents, so Palatino's here; symbols
        # Pagella lacks fall back to STIX
        "mathtext.fontset": "custom",
        "mathtext.rm": MATH,
        "mathtext.it": f"{MATH}:italic",
        "mathtext.bf": f"{MATH}:bold",
        "mathtext.bfit": f"{MATH}:bold:italic",
        "mathtext.tt": MONO,
        "mathtext.fallback": "stix",
    })
    return True

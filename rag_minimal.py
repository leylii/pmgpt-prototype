# rag_minimal.py — RAP evidence pipeline (sections + features + rule-based calibration)
# -----------------------------------------------------------------------------
# What this file does:
# 1) Build: index PDFs/TXTs, chunk text, embed with Sentence-Transformers, build FAISS
# 2) Tag each chunk into {requirements, design, implementation}
# 3) Extract features per chunk (screens, apis, integrations, etc.)
#    and compute document-level size separately from total word count
# 4) Calibrate using a RULE-BASED system (honest, explainable, works with few docs)
#    - Detects project size (small/medium/large) from total pages
#    - Applies complexity multipliers from detected signals (screens, apis, integrations)
#    - If rap_labels.json exists with 8+ projects, switches to data-driven base rates
# 5) Query: semantic search + numeric hour estimates per section
#
# Rule-based defaults (student projects, 4-person team, ~14 weeks):
#   small  → req: 25h,  design: 40h, impl: 80h
#   medium → req: 40h,  design: 65h, impl: 120h
#   large  → req: 55h, design: 90h, impl: 160h
#
# External deps: pypdf, sentence-transformers, faiss-cpu, numpy
# Optional: scikit-learn (only used when 8+ labeled projects exist)
# -----------------------------------------------------------------------------

import argparse, os, json, re, pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np


_MODEL = None

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ----------------------------- Rule-Based Config -----------------------------
# These are the core rules. Edit these numbers if your domain changes.
# Based on: Norwegian student software projects, ~4 person teams, ~14 week semester.

RULE_BASE_HOURS: Dict[str, Dict[str, float]] = {
    "small": {
        "requirements": 25.0,
        "design": 40.0,
        "implementation": 80.0
    },
    "medium": {
        "requirements": 40.0,
        "design": 65.0,
        "implementation": 120.0
    },
    "large": {
        "requirements": 55.0,
        "design": 90.0,
        "implementation": 160.0
    }
}

# How many total document pages define each project size
SIZE_THRESHOLDS = {"small": 10.0, "medium": 20.0}  # <10 = small, <20 = medium, else large

# Complexity signals: if a signal exceeds its threshold, add a bonus multiplier
COMPLEXITY_SIGNALS = [
    # (feature_key,  threshold,  bonus_multiplier)
    ("screens",      10,         0.15),   # many UI screens = more design/impl work
    ("integrations",  5,         0.20),   # external integrations are expensive
    ("apis",          5,         0.10),   # many API endpoints = more backend work
]

# Hard cap on total complexity bonus (so we never produce absurd numbers)
MAX_COMPLEXITY_MULTIPLIER = 1.5

# Minimum labeled projects needed before we trust data-driven base rates
MIN_LABELS_FOR_DATA_DRIVEN = 8

# -----------------------------------------------------------------------------


def _get_st_model():
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("PYTORCH_DEFAULT_DEVICE", "cpu")
        os.environ.setdefault("TRANSFORMERS_NO_META_DEVICE", "1")
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

# ----------------------------- Loaders -----------------------------

def load_txt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def load_pdf(fp: Path) -> str:
    """
    Robust PDF text extraction:
    1) PyMuPDF (fast)
    2) pdfplumber
    3) pypdf
    4) OCR fallback (pdf2image + pytesseract)
    """
    fp = Path(fp)
    texts = []

    def _norm(s: str) -> str:
        return (s or "").replace("\x00", "").strip()

    def _word_count(s: str) -> int:
        return len((s or "").split())

    MIN_WORDS_FOR_TEXT_LAYER = 120

    # 1) PyMuPDF
    try:
        import fitz
        doc = fitz.open(str(fp))
        for page in doc:
            texts.append(page.get_text("text") or "")
        out = _norm("\n".join(texts))
        if _word_count(out) >= MIN_WORDS_FOR_TEXT_LAYER:
            return out
    except Exception:
        pass

    # 2) pdfplumber
    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(str(fp)) as pdf:
            for p in pdf.pages:
                texts.append(p.extract_text() or "")
        out = _norm("\n".join(texts))
        if _word_count(out) >= MIN_WORDS_FOR_TEXT_LAYER:
            return out
    except Exception:
        pass

    # 3) pypdf
    try:
        from pypdf import PdfReader
        texts = []
        reader = PdfReader(str(fp))
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                texts.append("")
        out = _norm("\n".join(texts))
        if _word_count(out) >= MIN_WORDS_FOR_TEXT_LAYER:
            return out
    except Exception:
        pass

    # 4) OCR fallback
    try:
        print(f"[OCR] Using OCR for: {fp.name}")
        from pdf2image import convert_from_path
        import pytesseract
        lang = "eng"
        images = convert_from_path(str(fp), dpi=250)
        ocr_pages = []
        for img in images:
            ocr_pages.append(pytesseract.image_to_string(img, lang=lang) or "")
        out = _norm("\n".join(ocr_pages))
        return out
    except Exception as e:
        print(f"[OCR failed] {fp.name}: {e}")
        return ""

def load_file(fp: Path) -> str:
    if fp.suffix.lower() in [".txt", ".md"]:
        return load_txt(fp)
    if fp.suffix.lower() == ".pdf":
        return load_pdf(fp)
    raise ValueError(f"Unsupported file type: {fp.suffix} ({fp})")

# ----------------------------- Chunking -----------------------------

def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = 250, overlap: int = 60) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        piece = " ".join(words[i:j])
        if piece.strip():
            chunks.append({"text": normalize_ws(piece), "start_word": i, "end_word": j})
        if j == len(words):
            break
        i = max(j - overlap, 0)
    return chunks

def is_good_chunk(s: str) -> bool:
    if not s:
        return False
    w = s.split()
    if len(w) < 30:
        return False
    alpha = sum(ch.isalpha() for ch in s)
    if alpha / max(len(s), 1) < 0.35:
        return False
    return True

# -------------------- Section Tagging + Feature Extraction --------------------

SECTION_KEYS = {
    "requirements": [
        "requirement", "requirements", "user story", "user stories", "use case", "scope", "specification",
        "shall", "must", "should",
        "krav", "kravanalyse", "kravspesifikasjon",
        "brukerhistorie", "brukerhistorier",
        "interessent", "interessenter",
        "omfang", "prosjektmål", "mål",
        "skal", "må", "maa"
    ],
    "design": [
        "design", "architecture", "diagram", "uml", "sequence", "component", "class",
        "wireframe", "mockup", "prototype", "ui", "ux", "style guide", "figma", "erd", "figure", "fig.",
        "arkitektur", "systemarkitektur",
        "skisse", "prototyp", "prototyper",
        "nettsidedesign", "layout"
    ],
    "implementation": [
        "implement", "implementation", "implementering",
        "coding", "code", "koding",
        "frontend", "backend",
        "api", "endpoint", "route", "endepunkt",
        "database", "db",
        "deployment", "deploy", "hosting",
        "build",
        "testing", "test", "brukertesting", "systemtest", "enhetstest",
        "feilretting",
        "integration", "integrasjon",
        "nettsideutvikling", "integrering/implementering",
        "front-end", "back-end",
        "publisering", "lansering", "release",
        "bug", "fix",
        "integrasjonstest",
        "utvikling av nettside",
        "kode", "programmering",
        "implementere",
        "bygg", "bygge",
        "publisere",
    ]
}

FEATURE_WORDS = {
    "req_items": [
        "shall", "must", "should",
        "requirement", "requirements",
        "user story", "user stories",
        "use case", "use-case",
        "krav", "kravanalyse", "kravspesifikasjon",
        "brukerhistorie", "brukerhistorier",
        "skal", "må", "maa"
    ],
    "design_diagrams": [
        "diagram", "uml", "sequence", "component", "class", "activity",
        "figure", "fig.", "wireframe", "mockup", "erd", "prototype",
        "use-case-diagram", "use case diagram",
        "sekvensdiagram", "klassediagram", "komponentdiagram",
        "arkitektur", "systemarkitektur",
        "figma", "skisse"
    ],
    "screens": [
        "screen", "page", "view", "ui", "ux",
        "side", "sider", "skjerm", "skjermer",
        "forside", "startsiden", "startside",
        "prototype", "prototyp", "prototyper",
        "nettside", "nettsiden", "webside", "website",
        "booking", "booke", "reservasjon", "reservere",
        "landingsside"
    ],
    "apis": [
        "api", "apis", "endpoint", "endpoints", "route", "routes",
        "endepunkt", "endepunkter", "rute", "ruter"
    ],
    "integrations": [
        "integration", "integrations", "external", "oauth", "payment",
        "stripe", "github", "slack", "google",
        "integrasjon", "integrasjoner", "integrering",
        "kobling", "tilkobling", "tredjepart", "ekstern",
        "bookingsystem", "bookingsystemer",
        "figma", "discord",
        "publisering", "lansering"
    ],
    "quality_guidelines": [
        "code style", "lint", "coverage", "testing", "review", "guideline",
        "kvalitet", "retningslinje", "retningslinjer",
        "test", "testing", "brukertesting", "systemtest", "enhetstest",
        "feilretting", "kodegjennomgang",
        "bug", "bugs",
        "integrasjonstest",
        "deploy", "deployment",
        "publisere", "publisering",
        "lansering", "release",
        "bygg", "bygge", "build",
    ],
}

WORDS_PER_PAGE = 300


def guess_section(text: str) -> str:
    t = text.lower()
    scores = {k: 0 for k in SECTION_KEYS}
    for section, keys in SECTION_KEYS.items():
        scores[section] = count_kw(t, keys)
    best = max(scores, key=lambda k: scores[k])
    # Only return best if it clearly wins; otherwise default to implementation
    # (implementation is the largest phase in reality)
    return best if scores[best] > 0 else "implementation"

def count_kw(t: str, kws: List[str]) -> int:
    c = 0
    for kw in kws:
        if " " in kw or not kw[-1].isalnum():
            pattern = re.escape(kw)
        else:
            pattern = rf"\b{re.escape(kw)}\b"
        c += len(re.findall(pattern, t))
    return c
def extract_features(text: str) -> Dict[str, Any]:
    t = text.lower()

    f = {
        "req_items": 0,
        "design_diagrams": 0,
        "apis": 0,
        "screens": 0,
        "integrations": 0,
        "quality_guidelines": 0,
    }
    for key, kws in FEATURE_WORDS.items():
        f[key] = int(count_kw(t, kws))
    return f


# ----------------------------- Rule-Based Estimation -----------------------------

def detect_project_size(total_pages: float) -> str:
    """
    Classify project size based on total document pages.
    This is the simplest honest signal we have from the PDF.
    """
    if total_pages < SIZE_THRESHOLDS["small"]:
        return "small"
    elif total_pages < SIZE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "large"

def compute_complexity_multiplier(aggregated_features: Dict[str, float]) -> float:
    """
    Look at signals across the whole document and add bonuses for complexity.
    Each signal (many screens, many integrations, many APIs) adds a small bonus.
    Result is capped at MAX_COMPLEXITY_MULTIPLIER to avoid absurd estimates.
    """
    multiplier = 1.0
    for feat_key, threshold, bonus in COMPLEXITY_SIGNALS:
        if aggregated_features.get(feat_key, 0) > threshold:
            multiplier += bonus
    return min(multiplier, MAX_COMPLEXITY_MULTIPLIER)

def rule_based_estimate(section: str, total_pages: float, aggregated_features: Dict[str, float]) -> float:
    """
    Core rule-based estimator.
    1) Determine project size from total pages
    2) Look up base hours for that size and section
    3) Apply complexity multiplier
    Returns estimated hours for the given section.
    """
    size = detect_project_size(total_pages)
    base_hours = RULE_BASE_HOURS[size][section]
    multiplier = compute_complexity_multiplier(aggregated_features)
    return round(base_hours * multiplier, 1)

# ----------------------------- Build / Calibrate -----------------------------

def build_index(filepaths: List[str], outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path   = out / "meta.json"
    map_path    = out / "mapping.pkl"
    index_path  = out / "faiss.index"
    texts_path  = out / "texts.jsonl"
    feats_path  = out / "features.jsonl"

    print("Loading Sentence-Transformers (all-MiniLM-L6-v2)…")
    model = _get_st_model()

    texts, metadatas, mapping, features = [], [], [], []
    doc_id = 0
    for fp in filepaths:
        fp = Path(fp)
        print(f"Reading: {fp}")
        raw = load_file(fp)
        # --- compute real document pages ---
        doc_words = len(raw.split())
        doc_pages = doc_words / WORDS_PER_PAGE
        chunks = chunk_text(raw)
        chunks = [ch for ch in chunks if is_good_chunk(ch.get("text", ""))]
        for cid, ch in enumerate(chunks):
            txt = ch["text"]
            section = guess_section(txt)
            feats = extract_features(txt)
            texts.append(txt)
            metadatas.append({
                "source": fp.name,
                "doc_id": doc_id,
                "chunk_id": cid,
                "section": section,
                "doc_pages": round(doc_pages, 2)
            })
            mapping.append((doc_id, cid))
            features.append({
                "source": fp.name,
                "doc_id": doc_id,
                "chunk_id": cid,
                "section": section,
                "doc_pages": round(doc_pages, 2),
                **feats
            })
        doc_id += 1

    if not texts:
        raise RuntimeError("No text extracted — check your input files.")

    print(f"Embedding {len(texts)} chunks…")
    X = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    X = np.asarray(X, dtype="float32")

    print("Building FAISS index…")
    import faiss
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    print(f"Saving index → {index_path}")
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(metadatas, ensure_ascii=False, indent=2), encoding="utf-8")
    with map_path.open("wb") as f:
        pickle.dump(mapping, f)
    with texts_path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with feats_path.open("w", encoding="utf-8") as f:
        for row in features:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    maybe_calibrate(out)
    print("Done ✅")


def _aggregate_features_per_doc(feats_path: Path) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    with feats_path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            src = o["source"]
            a = agg.setdefault(src, {})
            for k, v in o.items():
                if k in {"source", "doc_id", "chunk_id", "section"}:
                    continue
                if k == "doc_pages":
                    a["approx_pages"] = float(v if v is not None else 0.0)
                    continue
                if k == "approx_pages":
                    continue
                try:
                    a[k] = a.get(k, 0.0) + float(v if v is not None else 0.0)
                except (TypeError, ValueError):
                    pass
    return agg


def maybe_calibrate(out: Path):
    """
    Build calibration.json using a rule-based approach.

    Two modes:
    - RULE-BASED (default, always safe): uses RULE_BASE_HOURS + complexity multipliers.
      Works with any number of documents. Honest and explainable.
    - DATA-DRIVEN (optional upgrade): if rap_labels.json exists AND has 8+ labeled projects,
      we compute real hours-per-page base rates from your data.
      This replaces the rule-based defaults with learned values.
    """
    labels_path = out / "rap_labels.json"
    feats_path  = out / "features.jsonl"
    calib_path  = out / "calibration.json"

    if not feats_path.exists():
        print("[calibrate] No features.jsonl found, skipping.")
        return

    # Aggregate features per document (whole-doc view)
    doc_features = _aggregate_features_per_doc(feats_path)

    print("\n[calibrate] Document summary:")
    for src, ft in sorted(doc_features.items()):
        total_pages = ft.get("approx_pages", 0.0)
        size = detect_project_size(total_pages)
        mult = compute_complexity_multiplier(ft)
        print(f"  {src}: size={size}, pages={total_pages:.1f}, complexity_mult={mult:.2f}, "
              f"screens={ft.get('screens',0):.0f}, apis={ft.get('apis',0):.0f}, "
              f"integrations={ft.get('integrations',0):.0f}")

    # Load labels if available
    labels: Dict[str, Any] = {}
    if labels_path.exists():
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            print(f"[calibrate] Loaded {len(labels)} labeled projects from rap_labels.json")
        except Exception as e:
            print(f"[calibrate] Could not read rap_labels.json: {e}")

    # Decide which mode to use
    labeled_count = sum(1 for src in labels if src in doc_features)
    use_data_driven = labeled_count >= MIN_LABELS_FOR_DATA_DRIVEN
    #use_data_driven = True

    if use_data_driven:
        print(f"[calibrate] {labeled_count} labeled projects found — using DATA-DRIVEN base rates.")
        base_rates = _compute_data_driven_base_rates(doc_features, labels)
        mode = "data_driven"
        mode_note = f"Data-driven base rates from {labeled_count} labeled projects (hours per page)."
    else:
        print(f"[calibrate] Only {labeled_count} labeled projects "
              f"(need {MIN_LABELS_FOR_DATA_DRIVEN}+) — using RULE-BASED defaults.")
        # In rule-based mode, base_rates are not used for estimation
        # (rule_based_estimate() is called directly at query time instead)
        # We store the rule table in calibration.json for transparency.
        base_rates = {}
        mode = "rule_based"
        mode_note = (
            f"Rule-based estimation (only {labeled_count} labeled projects available, "
            f"need {MIN_LABELS_FOR_DATA_DRIVEN}+ for data-driven mode). "
            f"Edit RULE_BASE_HOURS in rag_minimal.py to tune defaults."
        )

    calib = {
        "mode": mode,
        "base_rates": base_rates,  # only populated in data_driven mode
        "rule_based_hours": RULE_BASE_HOURS,
        "size_thresholds": SIZE_THRESHOLDS,
        "complexity_signals": [
            {"feature": s[0], "threshold": s[1], "bonus": s[2]}
            for s in COMPLEXITY_SIGNALS
        ],
        "note": mode_note,
    }

    calib_path.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[calibrate] calibration.json saved (mode={mode})")

    # Print what estimates would look like for each indexed document
    print("\n[calibrate] Sample estimates for indexed documents:")
    for src, ft in sorted(doc_features.items()):
        total_pages = ft.get("approx_pages", 0.0)
        for section in ("requirements", "design", "implementation"):
            if mode == "data_driven":
                est = _data_driven_estimate(section, total_pages, ft, base_rates)
            else:
                est = rule_based_estimate(section, total_pages, ft)
            print(f"  {src} | {section}: ~{est}h")
    print()


def _compute_data_driven_base_rates(
    doc_features: Dict[str, Dict[str, float]],
    labels: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute honest hours-per-page base rates from labeled data.
    Only called when we have enough labeled projects (MIN_LABELS_FOR_DATA_DRIVEN).
    Formula: base_rate = sum(hours) / sum(pages) across all labeled docs.
    """
    base_rates: Dict[str, float] = {}

    for section in ("requirements", "design", "implementation"):
        total_hours = 0.0
        total_pages = 0.0

        for src, ft in doc_features.items():
            lbl = labels.get(src, {})
            hours = float(lbl.get(f"{section}_hours", 0.0))
            pages = float(ft.get("approx_pages", 0.0))
            if hours > 0 and pages > 0:
                total_hours += hours
                total_pages += pages
                print(f"  [data] {section} | {src}: {hours}h / {pages:.1f}pages")

        if total_pages > 0 and total_hours > 0:
            rate = total_hours / total_pages
            # Sanity clamp: reasonable h/page for student projects
            _min = {"requirements": 5.0,  "design": 8.0,  "implementation": 15.0}[section]
            _max = {"requirements": 60.0, "design": 80.0, "implementation": 150.0}[section]
            base_rates[section] = round(max(_min, min(rate, _max)), 2)
            print(f"  [data] {section} base_rate = {base_rates[section]} h/page "
                  f"(from {total_hours}h / {total_pages:.1f} pages)")
        else:
            # Fallback to rule-based medium defaults converted to h/page
            fallback = {"requirements": 7.0, "design": 11.0, "implementation": 22.0}
            base_rates[section] = fallback[section]
            print(f"  [data] {section}: no labeled data found, using fallback {base_rates[section]} h/page")

    return base_rates


def _data_driven_estimate(
    section: str,
    total_pages: float,
    aggregated_features: Dict[str, float],
    base_rates: Dict[str, float]
) -> float:
    """
    Data-driven estimate: hours = base_rate * pages * complexity_multiplier
    Only used when we have enough labeled data.
    """
    rate = base_rates.get(section, 10.0)
    pages = max(total_pages, 0.5)
    multiplier = compute_complexity_multiplier(aggregated_features)
    return round(rate * pages * multiplier, 1)

# ----------------------------- Query helpers -----------------------------

def _load_lines_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

# ----------------------------- RagIndex -----------------------------

class RagIndex:
    def __init__(self, outdir: str):
        self.out = Path(outdir)
        self.meta_path  = self.out / "meta.json"
        self.map_path   = self.out / "mapping.pkl"
        self.index_path = self.out / "faiss.index"
        self.texts_path = self.out / "texts.jsonl"
        self.feats_path = self.out / "features.jsonl"
        self.calib_path = self.out / "calibration.json"

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        self.model = _get_st_model()

        import faiss
        if not all(p.exists() for p in [self.meta_path, self.map_path,
                                         self.index_path, self.texts_path]):
            raise FileNotFoundError("Index files not found. Run the 'build' command first.")

        self.metadatas = json.loads(self.meta_path.read_text(encoding="utf-8"))
        with self.map_path.open("rb") as f:
            self.mapping = pickle.load(f)
        self.index = faiss.read_index(str(self.index_path))
        self.texts = [o.get("text", "") for o in _load_lines_jsonl(self.texts_path)]
        self.features = _load_lines_jsonl(self.feats_path) if self.feats_path.exists() else [{} for _ in self.texts]

        try:
            self.calibration = (
                json.loads(self.calib_path.read_text(encoding="utf-8"))
                if self.calib_path.exists() else {}
            )
        except Exception:
            self.calibration = {}

        # Pre-aggregate features per document for estimation
        self._doc_features: Dict[str, Dict[str, float]] = {}
        for feat_row in self.features:
            src = feat_row.get("source", "")
            if not src:
                continue
            a = self._doc_features.setdefault(src, {})
            for k, v in feat_row.items():
                if k in {"source", "doc_id", "chunk_id", "section"}:
                    continue
                if k == "doc_pages":
                    a["approx_pages"] = float(v if v is not None else 0.0)
                    continue
                if k == "approx_pages":
                    continue
                try:
                    a[k] = a.get(k, 0.0) + float(v if v is not None else 0.0)
                except (TypeError, ValueError):
                    pass

    def _estimate_for_source(self, source: str, section: str) -> Dict[str, Any]:
        """
        Estimate hours for a given document and section using calibration.
        Also returns the project-size basis used for the estimate.
        """
        calib = self.calibration or {}
        mode = calib.get("mode", "rule_based")
        doc_ft = self._doc_features.get(source, {})
        total_pages = float(doc_ft.get("approx_pages", 5.0))

        project_size = detect_project_size(total_pages)
        complexity_multiplier = round(compute_complexity_multiplier(doc_ft), 2)

        if mode == "data_driven":
            base_rates = calib.get("base_rates", {})
            rate = float(base_rates.get(section, 10.0))
            estimated_hours = _data_driven_estimate(section, total_pages, doc_ft, base_rates)

            return {
                "hours": estimated_hours,
                "section": section,
                "mode": mode,
                "project_size": project_size,
                "total_pages": round(total_pages, 2),
                "complexity_multiplier": complexity_multiplier,
                "base_rate_hours_per_page": round(rate, 2),
                "estimation_basis": (
                    f"Estimated from detected project size '{project_size}' "
                    f"({total_pages:.1f} pages) using data-driven base rate "
                    f"{rate:.2f} h/page and complexity multiplier {complexity_multiplier:.2f}."
                ),
            }

        # Rule-based mode
        rule_hours = calib.get("rule_based_hours", RULE_BASE_HOURS)
        fallback_section_base = RULE_BASE_HOURS["medium"].get(section, 120.0)
        base = float(
            rule_hours.get(project_size, RULE_BASE_HOURS["medium"]).get(section, fallback_section_base)
        )
        estimated_hours = round(base * complexity_multiplier, 1)

        return {
            "hours": estimated_hours,
            "section": section,
            "mode": mode,
            "project_size": project_size,
            "total_pages": round(total_pages, 2),
            "complexity_multiplier": complexity_multiplier,
            "base_hours_for_size": round(base, 2),
            "estimation_basis": (
                f"Estimated from detected project size '{project_size}' "
                f"({total_pages:.1f} pages), base {base:.1f}h for section '{section}', "
                f"with complexity multiplier {complexity_multiplier:.2f}."
            ),
        }

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search — returns top-k most similar chunks."""
        qv = self.model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        D, I = self.index.search(qv, k)
        hits = []
        for rank, idx in enumerate(I[0].tolist()):
            md = self.metadatas[idx]
            hits.append({
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "source": md["source"],
                "doc_id": md["doc_id"],
                "chunk_id": md["chunk_id"],
                "section": md.get("section"),
                "text": self.texts[idx] if idx < len(self.texts) else "",
                "features": self.features[idx] if idx < len(self.features) else {},
            })
        return hits

    def nearest_chunks_with_doc_estimates(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the nearest text chunks and attach document-level hour estimates
        for the source document of each hit.

        Each hit is chunk-level evidence, while the attached estimates are computed
        from aggregated source/document features for requirements, design, and implementation.
        """
        hits = self.search(query, k=k)
        calib_mode = (self.calibration or {}).get("mode", "rule_based")
        out = []
        for h in hits:
            source = h["source"]
            h2 = dict(h)
            h2["estimated_hours"] = {
                section: self._estimate_for_source(source, section)
                for section in ("requirements", "design", "implementation")
            }
            h2["calibration_mode"] = calib_mode
            out.append(h2)
        return out
# ----------------------------- CLI -----------------------------

def query_index(outdir: str, query: str, k: int = 5):
    rag = RagIndex(outdir)
    hits = rag.nearest_chunks_with_doc_estimates(query, k=k)

    print(f"\nTop {k} results for: '{query}'")
    print("=" * 60)
    print("Note: snippet evidence is chunk-level; hour estimates are computed from the full source document.")
    for h in hits:
        est = h.get("estimated_hours", {})
        req = est.get("requirements", {})
        des = est.get("design", {})
        imp = est.get("implementation", {})

        print(
            f"\n[{h['rank']}] {h['source']} | section={h.get('section','?')} | "
            f"similarity={h['score']:.3f} | mode={h.get('calibration_mode','?')}"
        )
        print(
            f"     Estimates → "
            f"requirements: {req.get('hours','?')}h | "
            f"design: {des.get('hours','?')}h | "
            f"implementation: {imp.get('hours','?')}h"
        )

        # Show estimation basis once (requirements is enough since size/pages are doc-level)
        print(
            f"     Project basis → "
            f"size={req.get('project_size','?')} | "
            f"pages={req.get('total_pages','?')} | "
            f"complexity={req.get('complexity_multiplier','?')}"
        )
        print(f"     Why: {req.get('estimation_basis','?')}")
        print(f"     Snippet: {h['text'][:300]}…")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="RAP evidence pipeline: embed + features + rule-based calibration + estimates"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Embed files and build FAISS index + features + calibration")
    b.add_argument("--outdir", required=True, help="Output directory for index files")
    b.add_argument("files", nargs="+", help="Files to index (pdf/txt)")

    q = sub.add_parser("query", help="Query the index and get hour estimates")
    q.add_argument("--outdir", required=True, help="Directory with index files")
    q.add_argument("--k", type=int, default=5, help="Top-K results")
    q.add_argument("query", help="Your search query")

    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args.files, args.outdir)
    elif args.cmd == "query":
        query_index(args.outdir, args.query, args.k)

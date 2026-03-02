import re
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from dotenv import load_dotenv
import os
import importlib

def remove_small_components(bw: np.ndarray, keep_top: int = 12) -> np.ndarray:
    """
    bw must be binary: digits/noise in BLACK(0), background WHITE(255).
    Keeps only the largest connected components (digits are large; noise is small).
    """
    fg = (bw == 0).astype(np.uint8)  # 1 = ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return bw

    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    comps.sort(key=lambda x: x[1], reverse=True)
    keep_ids = set(i for i, _ in comps[:keep_top])

    cleaned_fg = np.zeros_like(fg)
    for i in keep_ids:
        cleaned_fg[labels == i] = 1

    cleaned_bw = np.where(cleaned_fg == 1, 0, 255).astype(np.uint8)
    return cleaned_bw


def autocrop_ink(bw: np.ndarray, pad: int = 15) -> np.ndarray:
    """
    Crops to bounding box of black pixels (ink), with padding.
    """
    ys, xs = np.where(bw == 0)
    if len(xs) == 0:
        return bw

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(bw.shape[1] - 1, x2 + pad)
    y2 = min(bw.shape[0] - 1, y2 + pad)

    return bw[y1:y2 + 1, x1:x2 + 1]

load_dotenv()
tcmd = os.getenv("TESSERACT_CMD", "").strip().strip('"')
pytesseract.pytesseract.tesseract_cmd = tcmd
print("Tesseract cmd =", pytesseract.pytesseract.tesseract_cmd)

_EASYOCR_READER = None

def get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER

    if importlib.util.find_spec("easyocr") is None:
        raise RuntimeError("EasyOCR not installed. Run: pip install easyocr")

    import easyocr
    # digits-only: language doesn't matter much, but 'en' is standard
    _EASYOCR_READER = easyocr.Reader(['en'], gpu=False)
    return _EASYOCR_READER

# Point this to your local Tesseract install on Windows:
# Example:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_for_digits(bgr: np.ndarray, debug: bool = False, debug_prefix: str = "dbg") -> np.ndarray:
    # 0) upscale pour faciliter la segmentation + OCR
    bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 1) HSV pour isoler le vert
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 2) Plage de vert (à tuner si besoin)
    # H: ~35-90 pour vert, S/V assez hauts pour “vert bien visible”
    lower = np.array([35, 60, 60], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)  # 255 là où c’est vert

    # 3) Nettoyage du masque (enlève speckles, comble petits trous)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) Convertir en "bw" attendu par tes helpers : digits en noir (0), fond blanc (255)
    bw = cv2.bitwise_not(mask)

    if debug:
        cv2.imwrite(f"{debug_prefix}_green_mask.png", mask)
        cv2.imwrite(f"{debug_prefix}_bw_from_green.png", bw)

    return bw

    
def read_10digits_easyocr_from_bw(bw: np.ndarray, debug: bool = True) -> str | None:
    """
    bw: image binaire/grayscale centrée sur les digits (idéalement).
    Retourne une séquence de 10 chiffres si trouvée.
    """
    reader = get_easyocr_reader()

    # EasyOCR attend souvent une image 3 canaux ; on convertit si besoin
    if len(bw.shape) == 2:
        img_for_ocr = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    else:
        img_for_ocr = bw

    # detail=0 -> liste de strings, allowlist -> digits only :contentReference[oaicite:3]{index=3}
    parts = reader.readtext(img_for_ocr, detail=0, allowlist="0123456789")
    if debug:
        print("EasyOCR parts:", parts)

    digits = re.sub(r"\D", "", "".join(parts))
    m = re.search(r"(\d{5})", digits)
    return m.group(1) if m else None

def read_invoice_5digits(image_path: str, debug: bool = True) -> str | None:
    p = Path(image_path)
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {p}")

    bw = preprocess_for_digits(bgr)

    if debug:
        cv2.imwrite(str(p.with_name(p.stem + "_preprocessed.png")), bw)

    # nettoyage + crop (ce que tu fais déjà)
    bw2 = remove_small_components(bw, keep_top=12)
    bw2 = autocrop_ink(bw2, pad=20)

    if debug:
        cv2.imwrite(str(p.with_name(p.stem + "_cleaned.png")), bw2)

    # 1) TESSERACT d'abord (rapide)
    best_digits = ""
    for psm in [7, 8, 13, 6]:
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
        raw = pytesseract.image_to_string(bw2, config=config)
        digits_only = re.sub(r"\D", "", raw)
        if debug:
            print(f"Tesseract PSM {psm} raw:", repr(raw), "digits:", digits_only)

        if len(digits_only) > len(best_digits):
            best_digits = digits_only

    m = re.search(r"(\d{5})", best_digits)
    if m:
        return m.group(1)

    # 2) FALLBACK EasyOCR (plus puissant)
    try:
        inv = read_10digits_easyocr_from_bw(bw2, debug=debug)
        if inv:
            return inv
    except Exception as e:
        if debug:
            print("EasyOCR failed:", e)

    return None
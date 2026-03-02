import re
import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv, find_dotenv



def _keep_largest_components(mask255: np.ndarray, keep_top: int = 8, min_area: int = 200) -> np.ndarray:
    """mask255: 0/255, garde les plus grosses composantes (les chiffres)."""
    fg = (mask255 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return mask255

    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    comps = [c for c in comps if c[1] >= min_area]
    comps.sort(key=lambda x: x[1], reverse=True)
    keep_ids = set(i for i, _ in comps[:keep_top])

    out = np.zeros_like(fg)
    for i in keep_ids:
        out[labels == i] = 1
    return (out * 255).astype(np.uint8)

def _autocrop_mask(mask255: np.ndarray, pad: int = 20) -> np.ndarray:
    ys, xs = np.where(mask255 > 0)
    if len(xs) == 0:
        return mask255
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(mask255.shape[1]-1, x2 + pad); y2 = min(mask255.shape[0]-1, y2 + pad)
    return mask255[y1:y2+1, x1:x2+1]

def preprocess_green_digits(bgr: np.ndarray, debug_prefix: str | None = None) -> np.ndarray:
    # Upscale: digits very large => OCR more stable
    bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Vert olive ~ #408000 : H autour de ~45-75 selon rendu.
    # On élargit un peu pour prendre les bords plus clairs.
    lower = np.array([30, 25, 40], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)  # 255 = vert

    # Enlève points isolés dans le masque
    mask = cv2.medianBlur(mask, 3)

    # Rebouche les trous causés par les points noirs sur les chiffres
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Optionnel: petit OPEN pour retirer mini artefacts verts
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # Ne garder que les grosses composantes (chiffres)
    mask = _keep_largest_components(mask, keep_top=10, min_area=400)

    # Crop autour des chiffres
    mask = _autocrop_mask(mask, pad=25)

    # Image finale pour Tesseract : chiffres NOIRS sur fond BLANC
    bw = cv2.bitwise_not(mask)

    # Ajoute une marge blanche (Tesseract aime bien)
    bw = cv2.copyMakeBorder(bw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_mask_green.png", mask)
        cv2.imwrite(f"{debug_prefix}_bw_final.png", bw)

    return bw

def read_invoice_5digits(image_path: str, debug: bool = True) -> str | None:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)

    bw = preprocess_green_digits(bgr, debug_prefix="debug" if debug else None)

    # psm 8 (single word) est souvent le meilleur pour 5 digits “collés”
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1 -c user_defined_dpi=300"
    raw = pytesseract.image_to_string(bw, config=config)
    digits = re.sub(r"\D", "", raw)

    m = re.search(r"\d{5}", digits)
    return m.group(0) if m else None
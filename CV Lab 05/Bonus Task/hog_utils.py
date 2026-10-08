"""HOG + SVM utilities for Lab 05 (GPU accelerated with PyTorch)."""
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_gray(img, size=128):
    """BGR/gray uint8 image -> grayscale uint8 (size x size)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def _hog_batch(g, cell, orient, clip, eps):
    """g: (N,H,W) float in [0,1] -> (N, D) HOG features (2x2 blocks, stride 1 cell, L2-Hys)."""
    N, H, W = g.shape
    g = g.unsqueeze(1)
    gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
    gx = gp[:, :, 1:-1, 2:] - gp[:, :, 1:-1, :-2]
    gy = gp[:, :, 2:, 1:-1] - gp[:, :, :-2, 1:-1]
    mag = torch.sqrt(gx * gx + gy * gy)
    ang = torch.atan2(gy, gx) % math.pi                  # unsigned orientation [0, pi)
    pos = ang / math.pi * orient - 0.5                    # soft-binning (linear interpolation)
    lo = torch.floor(pos)
    w_hi = pos - lo
    lo_i = lo.long() % orient
    hi_i = (lo.long() + 1) % orient
    hist = torch.zeros(N, orient, H, W, device=g.device, dtype=g.dtype)
    hist.scatter_add_(1, lo_i, mag * (1 - w_hi))
    hist.scatter_add_(1, hi_i, mag * w_hi)
    Hc, Wc = H // cell, W // cell
    hist = hist[:, :, : Hc * cell, : Wc * cell]
    cells = F.avg_pool2d(hist, cell, cell) * (cell * cell)   # (N, O, Hc, Wc) sum per cell
    blocks = torch.stack(
        [cells[:, :, :-1, :-1], cells[:, :, :-1, 1:], cells[:, :, 1:, :-1], cells[:, :, 1:, 1:]], dim=2
    )                                                      # (N, O, 4, Hc-1, Wc-1)
    blocks = blocks.permute(0, 3, 4, 2, 1).reshape(N, Hc - 1, Wc - 1, 4 * orient)
    blocks = blocks / torch.sqrt((blocks ** 2).sum(-1, keepdim=True) + eps ** 2)
    blocks = blocks.clamp(max=clip)
    blocks = blocks / torch.sqrt((blocks ** 2).sum(-1, keepdim=True) + eps ** 2)
    return blocks.reshape(N, -1)


@torch.no_grad()
def hog_torch(imgs, cell=8, orientations=9, clip=0.2, eps=1e-5, batch=128, device=None):
    """imgs: (N,H,W) or (H,W) array/tensor, pixel range 0..255. Returns float32 numpy (N, D)."""
    device = device or get_device()
    x = torch.as_tensor(imgs)
    if x.ndim == 2:
        x = x[None]
    out = []
    for i in range(0, len(x), batch):
        b = x[i: i + batch].to(device).float() / 255.0
        out.append(_hog_batch(b, cell, orientations, clip, eps).cpu())
    return torch.cat(out).numpy()


class KernelSVC:
    """SVM whose Gram matrix is computed on the GPU, solved by libsvm (sklearn SVC, precomputed kernel).
    Same solution as SVC(kernel='rbf'/'linear') but much faster for high-dimensional HOG vectors."""

    def __init__(self, kernel="rbf", C=10.0, probability=False, class_weight="balanced", device=None, seed=0):
        self.kernel, self.C, self.probability = kernel, C, probability
        self.class_weight, self.device, self.seed = class_weight, device, seed

    def _gram(self, A, B):
        dev = torch.device(self.device) if self.device else get_device()
        A = torch.as_tensor(A, dtype=torch.float32, device=dev)
        B = torch.as_tensor(B, dtype=torch.float32, device=dev)
        if self.kernel == "linear":
            K = self.gamma_ * (A @ B.T)
        else:
            d2 = (A * A).sum(1, keepdim=True) + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
            K = torch.exp(-self.gamma_ * d2.clamp_min(0))
        return K.double().cpu().numpy()

    def fit(self, X, y):
        self.Xtr_ = np.ascontiguousarray(X, dtype=np.float32)
        dev = torch.device(self.device) if self.device else get_device()
        var = float(torch.as_tensor(self.Xtr_, device=dev).var())
        self.gamma_ = 1.0 / (self.Xtr_.shape[1] * max(var, 1e-12))
        K = self._gram(self.Xtr_, self.Xtr_)
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=self.probability,
                        class_weight=self.class_weight, cache_size=1000, random_state=self.seed)
        self.svc_.fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(self._gram(X, self.Xtr_))

    def predict(self, X):
        return self.svc_.predict(self._gram(X, self.Xtr_))

    def predict_proba(self, X):
        return self.svc_.predict_proba(self._gram(X, self.Xtr_))


def inspect_product(img, bundle, stride=32, top_k=3):
    """Industrial quality-control decision for one product image (full image or a single patch).

    bundle: dict(model, cell, orient, size, patch, thr, min_windows)
    Full images are scanned with PATCH x PATCH sliding windows (same scale as the training patches).
    The product is REJECTED if at least `min_windows` windows are classified defective (prob >= thr).
    """
    m, P, S = bundle["model"], bundle["patch"], bundle["size"]
    thr, minw = bundle["thr"], bundle["min_windows"]
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    if max(H, W) <= S:                                    # already a single patch
        wins = [preprocess_gray(g, S)]
        coords = [(0, 0, W, H)]
    else:
        ys = list(range(0, max(H - P, 0) + 1, stride))
        xs = list(range(0, max(W - P, 0) + 1, stride))
        if ys[-1] != H - P:
            ys.append(max(H - P, 0))
        if xs[-1] != W - P:
            xs.append(max(W - P, 0))
        wins, coords = [], []
        for y in ys:
            for x in xs:
                crop = g[y:y + P, x:x + P]
                wins.append(cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC))
                coords.append((x, y, P, P))
    feats = hog_torch(np.stack(wins), cell=bundle["cell"], orientations=bundle["orient"])
    p = m.predict_proba(feats)[:, list(m.classes_).index(1)]
    pos = p >= thr
    need = 1 if len(p) == 1 else minw
    defective = bool(pos.sum() >= need)
    if defective:
        conf = float(p[pos].mean())
    else:
        conf = float(1.0 - np.sort(p)[-top_k:].mean())
    return dict(defective=defective, prediction="DEFECTIVE" if defective else "NON-DEFECTIVE",
                action="REJECT PRODUCT" if defective else "ACCEPT PRODUCT",
                confidence=conf, probs=p, coords=coords, positive=pos)


def print_report(res):
    print("PRODUCT INSPECTION RESULT")
    print(f"Prediction: {res['prediction']}")
    print(f"Confidence: {res['confidence'] * 100:.1f}%")
    print(f"Action: {res['action']}")

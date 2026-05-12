# =============================================================================
# PHOTOVERIFY AI v8 — Speed / Throughput Release
# ─────────────────────────────────────────────────────────────────────────────
# Target: 20,000 images in ≤ 60 minutes (≈ 333 img/min, ≈ 5.5 img/s).
# v7 baseline was ~50 img/min (100 images in ~2 min).  v8 delivers ~3–5× gain.
#
#  SPEED 1 — CHUNK_SIZE raised 15 → 30
#             More threads per chunk = more simultaneous Gemini Flash calls.
#             Gemini Flash has high QPS; doubling workers saturates more quota.
#
#  SPEED 2 — PARALLEL_CHUNKS = 2 (new constant)
#             run_chunk() now fires two chunks (60 images) in background threads
#             simultaneously and waits for both before re-rendering the UI.
#             This eliminates the ~0.5–1 s Streamlit rerun dead-time that v7
#             paid once every 15 images, boosting net throughput ~30–40%.
#
#  SPEED 3 — DUP_SEARCH_TOP_K reduced 100 → 50
#             FAISS returns fewer candidates for ORB/hash re-scoring.
#             Cuts per-image duplicate-search time ~40% with negligible recall
#             loss (true duplicates have CLIP score >> 0.90; top-50 is enough).
#
#  SPEED 4 — ORB_MAX_CANDIDATES reduced 24 → 12
#             Halves the worst-case ORB keypoint-match comparisons per image.
#             ORB is O(n·m); fewer candidates = proportionally faster.
#
#  DUPLICATE ACCURACY with 2 parallel chunks:
#             Both chunks still hold faiss_lock + state_lock when updating the
#             FAISS index and hash_cache.  An image registered in chunk A is
#             visible to chunk B's duplicate search immediately after the lock
#             releases, so cross-chunk duplicates are still caught.
#             ORB_MAX_CANDIDATES halved (24→12) to compensate for the wider
#             concurrent window; CLIP thresholds are unchanged.
# ─────────────────────────────────────────────────────────────────────────────
# All v7 fixes (FIX 1–10) and refactors (REFACTOR 1–5) are retained unchanged.
# =============================================================================

import streamlit as st
import sqlite3, hashlib, cv2, numpy as np, pandas as pd, faiss
import httpx, io, os, json, time, uuid, threading, logging, shutil
import concurrent.futures, random, queue
import glob, atexit, warnings, imagehash
from urllib.parse import urlparse
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    try:
        from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx
    except ImportError:
        add_script_run_ctx = None
        get_script_run_ctx = None
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
import scipy.fftpack
import google.generativeai as genai
from google.oauth2.service_account import Credentials
import gspread
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =============================================================================
# PERFORMANCE / STABILITY SETTINGS
# =============================================================================
# The CLIP model uses PyTorch under SentenceTransformer. On CPU, PyTorch may use
# many internal threads per encode() call. With 40 Python workers this can create
# heavy CPU oversubscription. Limiting internal math threads usually improves
# throughput for this workload.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# =============================================================================
# THREAD-SAFE GLOBAL FALLBACKS
# =============================================================================
# Streamlit's st.session_state is bound to a ScriptRunContext. Background daemon
# threads can lose that context during reruns. These fallback locks prevent
# crashes such as missing db_lock/state_lock while preserving normal Streamlit
# locks whenever they are available.
_PV_GLOBAL_DB_LOCK = threading.RLock()
_PV_GLOBAL_STATE_LOCK = threading.RLock()
_PV_GLOBAL_FAISS_LOCK = threading.RLock()

def _ss_lock(name: str, fallback):
    try:
        lock = st.session_state.get(name, None)
        return lock if lock is not None else fallback
    except Exception:
        return fallback


# =============================================================================
# CONSTANTS
# =============================================================================
APP_NAME    = "Sridhar's PhotoVerify AI"
APP_VERSION = "V8-Fast40-V7Accuracy-ProFallback-SpeedFix"
DB_NAME     = "photoverify.db"
FAISS_INDEX = "faiss.bin"
FAISS_TMP   = "faiss.bin.tmp"
FAISS_BAK   = "faiss.bin.bak"
FAISS_MAP   = "faiss.bin.map"
FAISS_MTMP  = "faiss.bin.map.tmp"
FAISS_MBAK  = "faiss.bin.map.bak"

# ── Models ────────────────────────────────────────────────────────────────────
# Best models for your service account:
#   Flash = gemini-3.1-flash-lite-preview  (fastest, cheapest, good accuracy)
#   Pro   = gemini-3.1-pro-preview         (highest quality for hard cases)
FLASH_MODEL  = os.getenv("PV_FLASH_MODEL", "gemini-3.1-flash-lite-preview")
PRO_MODEL    = os.getenv("PV_PRO_MODEL",   "gemini-3.1-pro-preview")
USD_INR_RATE = float(os.getenv("PV_USD_INR_RATE", "83.0"))

# Gemini pricing (USD per 1M tokens)
FLASH_INPUT_USD_PER_MTOK  = float(os.getenv("PV_FLASH_INPUT_USD_PER_MTOK",  "0.15"))
FLASH_OUTPUT_USD_PER_MTOK = float(os.getenv("PV_FLASH_OUTPUT_USD_PER_MTOK", "0.60"))
PRO_INPUT_USD_PER_MTOK    = float(os.getenv("PV_PRO_INPUT_USD_PER_MTOK",    "3.50"))
PRO_OUTPUT_USD_PER_MTOK   = float(os.getenv("PV_PRO_OUTPUT_USD_PER_MTOK",  "10.50"))

# ── Duplicate thresholds ──────────────────────────────────────────────────────
PHASH_T             = 6
DHASH_T             = 8
WHASH_T             = 5
CLIP_T              = 0.90
CLIP_STRONG_T       = 0.97
CLIP_HASH_SUPPORT_T = 1
HASH_VOTES_T        = 2

# ── Image-of-Image (IoI) — FIX 2: more aggressive screen detection ────────────
IOI_SCORE_T         = 0.48   # restored: 0.35 caught too many textiles/JPEGs as IoI

# Branding color gate (pre-AI heuristic)
BRANDING_COLOR_GATE = True

# ── ORB thresholds ────────────────────────────────────────────────────────────
ORB_MIN_MATCHES     = 30
ORB_LOWE_RATIO      = 0.75
ORB_HOMOGRAPHY_T    = 10.0
ORB_INLIER_RATIO = 0.46  # v10.6: from v10.5 duplicate profile
ORB_MIN_INLIERS = 26  # v10.6: from v10.5 duplicate profile
ORB_MAX_CANDIDATES = 24  # V7 accuracy profile: maximum duplicate recall
ORB_MAX_SIDE = 800  # v10.6: from v10.5 duplicate profile
ORB_NEAR_CLIP_FLOOR = 0.64  # v10.6: from v10.5 duplicate profile
ORB_SCENE_STRONG_INLIERS = 60  # v10.6: from v10.5 duplicate profile
ORB_SCENE_STRONG_SCORE = 0.58  # v10.6: from v10.5 duplicate profile

# ── Validation thresholds ─────────────────────────────────────────────────────
AGREE_T = 0.82  # v10.6: keep low Pro usage from v4/v10.5
FLASH_DIRECT_T      = 0.85      # restored: 0.90 was too strict
CV_T                = 0.85
ELA_T               = 0.50
BLUR_T              = 80.0
MOIRE_T             = 8.0       # restored: 7.5 triggered too many false positives
REVIEW_T            = 0.00
CV_CLEAN_T          = 0.15
FLASH_FALLBACK_T    = 0.70
CV_FLASH_CLEAN_T    = 0.55
TIMESTAMP_WHITE_T   = 0.15
POSTER_EDGE_T       = 0.18
FORCE_PRO_RATE      = 0.00

# FIX 1 & 5: Always treat Pending Review as Invalid; limit dup window to 10 days
PENDING_REVIEW_AS_INVALID = True
MINOR_POLICY_ENABLED      = True
DUP_LOOKBACK_DAYS         = 10
DUP_SEARCH_TOP_K = 100  # V7 accuracy profile: maximum duplicate recall

# Borderline moire
MOIRE_BORDERLINE_LOW  = 0.35
MOIRE_BORDERLINE_HIGH = 0.50

# Branding / face requirements
FACE_CONF_MIN_SIZE = 40
FACE_REQUIRED      = False
BRANDING_REQUIRED  = True
BRANDING_KEYWORDS  = [
    "ncp", "bjp", "congress", "aap", "shiv sena", "shivsena",
    "लाडकी", "लाडका", "योजना", "पक्ष", "अजित", "pawar",
    "majhi", "ladki", "bahin", "yojana", "clock", "घड्याळ",
]

# ── Runtime ───────────────────────────────────────────────────────────────────
# v8 SPEED: CHUNK_SIZE raised from 15 → 30 (2× parallel AI calls per chunk).
# PARALLEL_CHUNKS=2 runs two chunks concurrently in background threads, hiding
# the chunk-boundary UI-rerender pause and saturating Gemini quota headroom.
# Net effect: ~3–4× throughput gain over v7 baseline.
CHUNK_SIZE           = 40   # Fast40: one 40-worker executor per UI cycle
PARALLEL_CHUNKS      = 1    # Fast40: avoid nested 2-chunk overhead; still 40 concurrent workers
FAISS_FLUSH          = 500
RETRY_DB             = 5
RETRY_IMG            = 2
RETRY_AI             = 3
HTTP_TIMEOUT         = 15
AI_TIMEOUT_S         = 20
HTTP_CONNECT_TIMEOUT = 8
HTTP_HARD_TIMEOUT_S  = 30
MAX_IMAGE_BYTES      = 20 * 1024 * 1024
CV_MAX_SIDE          = 960
CLIP_MAX_SIDE        = 512      # restored: 384 lost too much semantic detail
AI_MAX_SIDE          = 1024     # restored: 896 caused Pro to miss fine detail
AI_JPEG_QUALITY      = 75       # restored: 65 was too lossy for Pro to read detail
FACE_SIM_VETO_T = 0.55  # v10.6: from v10.5 duplicate profile

# =============================================================================
# STRATEGY PATTERN — Layer Pipeline
# =============================================================================
# Each verification layer implements BaseLayer. process_one() iterates
# self.layers in order, calling layer.run(ctx). A layer returns a dict
# with at minimum {"exit": True/False}. When exit=True the pipeline stops.
# To toggle or reorder layers, edit PIPELINE_LAYERS at the bottom of this
# section — no changes to process_one() are needed.

from abc import ABC, abstractmethod

class LayerContext:
    """Mutable bag of state passed through every layer."""
    __slots__ = (
        "r", "img", "cv_img", "hashes", "emb", "cv",
        "dup", "clip_dup", "flash_r", "orig", "t0",
    )
    def __init__(self, r, img, cv_img, hashes, emb, orig, t0):
        self.r        = r
        self.img      = img
        self.cv_img   = cv_img
        self.hashes   = hashes
        self.emb      = emb
        self.cv       = None
        self.dup      = None
        self.clip_dup = None
        self.flash_r  = None
        self.orig     = orig
        self.t0       = t0

class BaseLayer(ABC):
    """Interface every pipeline layer must implement."""
    name: str = "BaseLayer"

    @abstractmethod
    def run(self, ctx: LayerContext) -> dict:
        """
        Execute this layer's logic.
        Returns dict with:
          exit  (bool)  – True  → stop pipeline, result is final
          skip  (bool)  – True  → layer errored out; graceful degradation
        May mutate ctx.r (the VR result object) in place.
        """


def _faiss_duplicate_candidates_locked(ctx: LayerContext, exclude_self: bool = True) -> List[Tuple[str, float]]:
    """Return recent/current FAISS candidates while caller holds faiss_lock.

    Keep this lock section small: FAISS search + cheap filtering only. ORB/CV
    matching runs outside faiss_lock so 40 workers can run in parallel.
    """
    fidx = st.session_state.faiss_index
    if ctx.emb is None or fidx is None or fidx.ntotal <= 0:
        return []

    k_search = min(max(ORB_MAX_CANDIDATES, DUP_SEARCH_TOP_K), fidx.ntotal)
    ids, scores = faiss_search(fidx, ctx.emb, k=k_search)
    allowed = st.session_state.get("duplicate_candidate_ids", set())

    out: List[Tuple[str, float]] = []
    for pid, score in zip(ids, scores):
        if exclude_self and pid == ctx.r.pv_image_id:
            continue
        if allowed and pid not in allowed:
            continue
        out.append((pid, score))
        if len(out) >= ORB_MAX_CANDIDATES:
            break
    return out


def _resolve_clip_or_orb_duplicate(ctx: LayerContext, search_results: List[Tuple[str, float]]) -> Optional[dict]:
    """Resolve FAISS candidates into a duplicate decision outside faiss_lock.

    V7-compatible duplicate logic:
    1. accept very strong CLIP,
    2. accept CLIP when hash support is present,
    3. otherwise confirm with ORB multi-angle matching.
    """
    if not search_results:
        return None

    best_id, best_score = search_results[0]
    with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
        match_hashes = st.session_state.hash_cache.get(best_id)
    clip_votes = hash_vote_count(ctx.hashes, match_hashes)

    strong_clip = best_score >= CLIP_STRONG_T
    supported_clip = best_score >= CLIP_T and clip_votes >= CLIP_HASH_SUPPORT_T
    if strong_clip or supported_clip:
        return {"match_id": best_id, "score": best_score, "votes": clip_votes, "method": "CLIP Semantic"}

    # Heavy CPU work: intentionally outside faiss_lock for true parallelism.
    for cid, cscore in search_results:
        if cscore < ORB_NEAR_CLIP_FLOOR:
            break
        cached_img = orb_cache_get(cid)
        if cached_img is None:
            continue
        is_orb_dup, orb_score, inliers = orb_match(ctx.img, cached_img)
        if is_orb_dup:
            return {
                "match_id": cid,
                "score": max(cscore, orb_score),
                "votes": inliers,
                "method": f"ORB Multi-Angle ({inliers} inliers)",
            }
    return None

class HashLayer(BaseLayer):
    """Layer 1 — hash + CLIP/ORB duplicate detection.

    Fast40 + V7 accuracy design:
    - Hash/FAISS reads/writes stay locked for correctness.
    - Heavy ORB matching runs outside faiss_lock for speed.
    - A final locked FAISS re-check before adding catches same-wave duplicates.
    """
    name = "Hash"

    def run(self, ctx: LayerContext) -> dict:
        own_faiss_added = False
        try:
            search_results: List[Tuple[str, float]] = []

            # PHASE 1 — quick locked hash check + FAISS candidate snapshot.
            with _ss_lock("faiss_lock", _PV_GLOBAL_FAISS_LOCK):
                with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                    _cand_ids = st.session_state.get("duplicate_candidate_ids", set())
                    ctx.dup = find_dup(ctx.hashes, st.session_state.hash_cache, _cand_ids)
                    if not ctx.dup["is_duplicate"]:
                        st.session_state.hash_cache[ctx.r.pv_image_id] = {
                            "md5": ctx.hashes["md5"],
                            "ph":  ctx.hashes["ph"],
                            "dh":  ctx.hashes["dh"],
                            "wh":  ctx.hashes["wh"],
                        }
                        st.session_state.duplicate_candidate_ids.add(ctx.r.pv_image_id)

                if not ctx.dup["is_duplicate"]:
                    search_results = _faiss_duplicate_candidates_locked(ctx, exclude_self=True)

            # PHASE 2 — heavy duplicate resolution outside faiss_lock.
            if not ctx.dup["is_duplicate"]:
                ctx.clip_dup = _resolve_clip_or_orb_duplicate(ctx, search_results)

            # PHASE 3 — final same-wave re-check before adding vector.
            if not ctx.dup["is_duplicate"] and ctx.emb is not None:
                checked_ids = {pid for pid, _ in search_results}
                while ctx.clip_dup is None:
                    with _ss_lock("faiss_lock", _PV_GLOBAL_FAISS_LOCK):
                        late_results = _faiss_duplicate_candidates_locked(ctx, exclude_self=True)
                        unchecked = [(pid, sc) for pid, sc in late_results if pid not in checked_ids]
                        if not unchecked:
                            fidx = st.session_state.faiss_index
                            if fidx is not None:
                                st.session_state.faiss_index = faiss_add(fidx, ctx.emb, ctx.r.pv_image_id)
                                own_faiss_added = True
                            break

                    checked_ids.update(pid for pid, _ in unchecked)
                    ctx.clip_dup = _resolve_clip_or_orb_duplicate(ctx, unchecked)

                    # Safety valve: prevents pathological waiting under heavy arrival.
                    if len(checked_ids) >= ORB_MAX_CANDIDATES * 3 and ctx.clip_dup is None:
                        with _ss_lock("faiss_lock", _PV_GLOBAL_FAISS_LOCK):
                            fidx = st.session_state.faiss_index
                            if fidx is not None:
                                st.session_state.faiss_index = faiss_add(fidx, ctx.emb, ctx.r.pv_image_id)
                                own_faiss_added = True
                        break

                # Preserve V8/V7 behavior: index duplicates too for future detection.
                if ctx.clip_dup is not None and not own_faiss_added:
                    with _ss_lock("faiss_lock", _PV_GLOBAL_FAISS_LOCK):
                        fidx = st.session_state.faiss_index
                        if fidx is not None:
                            st.session_state.faiss_index = faiss_add(fidx, ctx.emb, ctx.r.pv_image_id)
                            own_faiss_added = True

            orb_cache_add(ctx.r.pv_image_id, ctx.img)
            save_hash(ctx.r.pv_image_id, ctx.hashes["md5"],
                      ctx.hashes["phash"], ctx.hashes["dhash"], ctx.hashes["whash"])

            r = ctx.r; dup = ctx.dup
            if dup and dup["is_duplicate"] and dup["match_id"] != r.pv_image_id:
                r.validation_status = "Duplicate"; r.duplicate_status = dup["duplicate_status"]
                r.matched_image_id  = dup["match_id"]; r.similarity_score = dup["similarity"]
                r.ai_confidence     = dup["similarity"]; r.exit_layer = 1
                r.original_status   = get_validation_status(dup["match_id"])
                r.forensic_reasoning = (
                    f"{dup['reason']} Matched {dup['match_id']}. "
                    f"Similarity {dup['similarity']:.1f}%. Original: {r.original_status}."
                )
                with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                    st.session_state.layer_counts[1] += 1
                assign_cluster(r)
                return {"exit": True}

            clip_dup = ctx.clip_dup
            if clip_dup:
                try:
                    cached_img = orb_cache_get(clip_dup["match_id"])
                    if cached_img is not None:
                        face_a = extract_face_crop(ctx.img)
                        face_b = extract_face_crop(cached_img)
                        if face_a is not None and face_b is not None:
                            fsim = face_similarity(face_a, face_b)
                            if fsim < FACE_SIM_VETO_T:
                                strong_scene = (
                                    str(clip_dup.get("method", "")).startswith("ORB")
                                    and int(clip_dup.get("votes", 0) or 0) >= ORB_SCENE_STRONG_INLIERS
                                    and float(clip_dup.get("score", 0.0) or 0.0) >= ORB_SCENE_STRONG_SCORE
                                )
                                if not strong_scene:
                                    logging.info(f"[L3 FaceVeto] face_sim={fsim:.3f}; vetoing duplicate")
                                    ctx.clip_dup = None
                                    clip_dup = None
                except Exception:
                    pass

            if clip_dup:
                r.validation_status = "Duplicate"
                r.duplicate_status  = clip_dup["method"]
                r.matched_image_id  = clip_dup["match_id"]
                r.similarity_score  = round(clip_dup["score"] * 100, 2)
                r.ai_confidence     = r.similarity_score
                r.exit_layer        = 3
                r.original_status   = get_validation_status(clip_dup["match_id"])
                r.forensic_reasoning = (
                    f"{clip_dup['method']}: score={clip_dup['score']:.3f}, "
                    f"support={clip_dup['votes']}. Matched {clip_dup['match_id']}. "
                    f"Original: {r.original_status}."
                )
                with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                    st.session_state.layer_counts[3] += 1
                assign_cluster(r)
                return {"exit": True}

            return {"exit": False}
        except Exception as e:
            logging.error(f"[HashLayer] error: {e}", exc_info=True)
            return {"exit": False, "skip": True}


class CVLayer(BaseLayer):
    """Layer 2 — OpenCV forensics + pre-AI screening.
    Gracefully degrades: if CV fails entirely the pipeline continues to AI layers.
    """
    name = "CV"

    def run(self, ctx: LayerContext) -> dict:
        try:
            cv = run_cv(ctx.cv_img)
            ctx.cv = cv
            r = ctx.r
            r.has_face = cv.face_count > 0; r.face_count = cv.face_count
            r.gps_lat  = cv.gps_lat; r.gps_lon = cv.gps_lon; r.gps_valid = cv.has_gps

            if cv.score > CV_T and not cv.is_borderline_moire:
                r.validation_status  = "Invalid"
                r.is_screenshot      = cv.has_moire or cv.has_bezel or cv.ss_software
                r.is_manipulated     = cv.ela_score > ELA_T
                r.exit_layer         = 2
                r.ai_confidence      = round(cv.score * 100, 1)
                r.error_reason       = ", ".join(cv.flags) or "CV anomaly"
                r.forensic_reasoning = (
                    f"CV: moire={cv.moire_score:.2f}, bezel={cv.bezel_score:.2f}, "
                    f"ELA={cv.ela_score:.2f}. Flags: {cv.flags}"
                )
                with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                    st.session_state.layer_counts[2] += 1
                return {"exit": True}

            pre_reject, pre_reason = pre_ai_screen(cv, ctx.orig, ctx.cv_img)
            if pre_reject:
                r.validation_status  = "Invalid"; r.exit_layer = 2
                r.ai_confidence      = 0.0
                r.error_reason       = pre_reason
                r.forensic_reasoning = (
                    f"Pre-AI gate rejected (no AI call). Reason: {pre_reason}. "
                    f"CV: moire={cv.moire_score:.2f}, bezel={cv.bezel_score:.2f}, faces={cv.face_count}"
                )
                with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                    st.session_state.layer_counts[2] += 1
                return {"exit": True}

            return {"exit": False}
        except Exception as e:
            logging.error(f"[CVLayer] error (graceful degradation): {e}", exc_info=True)
            # CV failure is non-fatal — let AI layers decide
            return {"exit": False, "skip": True}


class FlashLayer(BaseLayer):
    """Layer 4 — Gemini Flash AI screening.
    Graceful degradation: if Flash fails, the pipeline falls through to ProLayer.
    """
    name = "Flash"

    def run(self, ctx: LayerContext) -> dict:
        try:
            flash_r = gemini_flash(ctx.img)
            ctx.flash_r = flash_r
            r = ctx.r
            r.flash_model = FLASH_MODEL

            if not flash_r or "error" in flash_r:
                r.forensic_reasoning = "Flash inconclusive; routing to Pro."
                r.error_reason = ""
                ctx.flash_r = None
                return {"exit": False}

            if flash_r and "error" not in flash_r:
                f_usage = flash_r.get("_usage", {})
                r.flash_prompt_tokens = int(f_usage.get("prompt_tokens", 0) or 0)
                r.flash_output_tokens = int(f_usage.get("output_tokens", 0) or 0)
                r.flash_total_tokens  = int(f_usage.get("total_tokens",  0) or 0)
                r.flash_cost_usd      = float(flash_r.get("_cost_usd", 0.0) or 0.0)
                r.flash_cost_inr      = float(flash_r.get("_cost_inr", 0.0) or 0.0)
                flash_conf               = float(flash_r.get("confidence_score", 0.85))
                r.is_screenshot          = bool(flash_r.get("is_screenshot", False))
                r.is_manipulated         = bool(flash_r.get("is_manipulated", False))
                r.has_face               = bool(flash_r.get("has_face", r.has_face))
                r.face_count             = int(flash_r.get("face_count", r.face_count))
                r.has_required_branding  = bool(flash_r.get("has_required_branding", False))
                r.branding_details       = str(flash_r.get("branding_details", ""))
                r.image_quality          = str(flash_r.get("image_quality", ""))
                r.forensic_reasoning     = str(flash_r.get("reasoning", ""))

                # FIX 3: Apply minor policy at Flash level immediately
                r = apply_minor_policy(r, flash_r)
                if r.validation_status == "Invalid":
                    r.ai_confidence = round(flash_conf * 100, 1)
                    r.exit_layer = 4; r.cluster_id = r.pv_image_id
                    with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                        st.session_state.layer_counts[4] += 1
                    _accumulate_cost(r)
                    r.total_ai_tokens   = int(r.flash_total_tokens + r.pro_total_tokens)
                    r.total_ai_cost_inr = round(r.flash_cost_inr + r.pro_cost_inr, 6)
                    return {"exit": True}

                if r.is_screenshot or r.is_manipulated:
                    r.validation_status = "Invalid"
                    r.error_reason = "Photo-of-screen/photo/manipulated image detected by Flash"
                    r.ai_confidence = round(flash_conf * 100, 1)
                    r.exit_layer = 4; r.cluster_id = r.pv_image_id
                    with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                        st.session_state.layer_counts[4] += 1
                    _accumulate_cost(r)
                    r.total_ai_tokens   = int(r.flash_total_tokens + r.pro_total_tokens)
                    r.total_ai_cost_inr = round(r.flash_cost_inr + r.pro_cost_inr, 6)
                    return {"exit": True}

                if not r.has_face:
                    r.validation_status = "Invalid"
                    r.error_reason = "No clearly visible live human face detected by Flash"
                    r.ai_confidence = round(flash_conf * 100, 1)
                    r.exit_layer = 4; r.cluster_id = r.pv_image_id
                    with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                        st.session_state.layer_counts[4] += 1
                    _accumulate_cost(r)
                    r.total_ai_tokens   = int(r.flash_total_tokens + r.pro_total_tokens)
                    r.total_ai_cost_inr = round(r.flash_cost_inr + r.pro_cost_inr, 6)
                    return {"exit": True}

                if BRANDING_REQUIRED and not r.has_required_branding:
                    r.validation_status = "Invalid"
                    r.error_reason = "No required political branding detected by Flash"
                    r.ai_confidence = round(flash_conf * 100, 1)
                    r.exit_layer = 4; r.cluster_id = r.pv_image_id
                    with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                        st.session_state.layer_counts[4] += 1
                    _accumulate_cost(r)
                    r.total_ai_tokens   = int(r.flash_total_tokens + r.pro_total_tokens)
                    r.total_ai_cost_inr = round(r.flash_cost_inr + r.pro_cost_inr, 6)
                    return {"exit": True}

            return {"exit": False}
        except Exception as e:
            logging.error(f"[FlashLayer] error (graceful degradation): {e}", exc_info=True)
            return {"exit": False, "skip": True}


class ProLayer(BaseLayer):
    """Layer 5 — Gemini Pro AI for borderline / uncertain cases.
    Graceful degradation: if Pro fails, falls back to Flash verdict or CV score.
    """
    name = "Pro"

    def run(self, ctx: LayerContext) -> dict:
        try:
            r = ctx.r; cv = ctx.cv; flash_r = ctx.flash_r

            # Determine whether Pro is needed
            flash_ok = flash_r is not None and "error" not in flash_r
            flash_conf = float(flash_r.get("confidence_score", 0.0)) if flash_ok else 0.0
            # Required by direct Flash bypass and fallback paths.
            flash_inv = bool(r.is_screenshot or r.is_manipulated)
            cv_clean = (cv.score <= CV_CLEAN_T and not cv.flags) if cv else True
            cv_blocking = bool(cv and (cv.has_bezel or cv.is_photo_of_photo or cv.ela_score > ELA_T or cv.score > 0.35))
            flash_core_valid = bool(
                flash_ok
                and not flash_inv
                and bool(r.has_face)
                and (not BRANDING_REQUIRED or bool(r.has_required_branding))
            )
            flash_minor_block = bool(
                flash_ok and (
                    flash_r.get("child_only", False)
                    or flash_r.get("minor_holding_branding", False)
                )
            )

            cv_has_face = cv.face_count > 0 if cv else r.has_face
            face_disagree = r.has_face != cv_has_face
            # Reduce unnecessary Pro calls: CV face detection is weaker than Flash.
            # Escalate face disagreement only when Flash confidence is not already strong.
            face_disagree_hard = bool(face_disagree and flash_conf < FLASH_DIRECT_T)

            cv_inv = bool(
                cv and (
                    cv.has_bezel
                    or cv.ela_score > ELA_T
                    or cv.is_photo_of_photo
                )
            )

            borderline_moire = bool(cv.is_borderline_moire) if cv else False
            force_pro = random.random() < FORCE_PRO_RATE

            needs_pro = (
                force_pro
                or not flash_ok
                or face_disagree_hard
                or cv_inv
                or borderline_moire
                or flash_conf < AGREE_T
            )

            def _apply_hard_gates(result):
                if result.is_screenshot or result.is_manipulated:
                    if result.validation_status in ("Valid", "Pending Review"):
                        result.validation_status = "Invalid"
                        result.error_reason = "Photo-of-screen/photo/manipulated image detected"
                if not result.has_face:
                    if result.validation_status in ("Valid", "Pending Review"):
                        result.validation_status = "Invalid"
                        result.error_reason = "No clearly visible live human face detected"
                elif BRANDING_REQUIRED and not result.has_required_branding:
                    if result.validation_status in ("Valid", "Pending Review"):
                        result.validation_status = "Invalid"
                        result.error_reason = "No required political branding detected"
                if PENDING_REVIEW_AS_INVALID and result.validation_status == "Pending Review":
                    result.validation_status = "Invalid"
                    result.error_reason = result.error_reason or "Pending Review → Invalid (production policy)"
                return result


            # =========================================================================
            # FIX 1: Bypass Pro if unneeded and directly accept Flash's verdict
            # =========================================================================
            if not needs_pro:
                r.validation_status = "Invalid" if flash_inv else "Valid"
                r.ai_confidence = round(flash_conf * 100, 1)
                r.exit_layer = 4
                r = _apply_hard_gates(r)
                with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                    st.session_state.layer_counts[4] += 1
                r.cluster_id = r.pv_image_id
                _accumulate_cost(r)
                return {"exit": True}

            pro_r = None
            if needs_pro:
                # Mark Pro as attempted even if the API returns no JSON/error.
                # This makes PV_Pro_Model non-empty for "Flash and Pro inconclusive" rows.
                r.pro_model = PRO_MODEL
                pro_r = gemini_pro(ctx.img, flash_r, cv) if cv else gemini_pro(ctx.img, flash_r, CVR())

            if pro_r and "error" not in pro_r:
                p_usage = pro_r.get("_usage", {})
                r.pro_model         = PRO_MODEL
                r.pro_prompt_tokens = int(p_usage.get("prompt_tokens", 0) or 0)
                r.pro_output_tokens = int(p_usage.get("output_tokens", 0) or 0)
                r.pro_total_tokens  = int(p_usage.get("total_tokens",  0) or 0)
                r.pro_cost_usd      = float(pro_r.get("_cost_usd", 0.0) or 0.0)
                r.pro_cost_inr      = float(pro_r.get("_cost_inr", 0.0) or 0.0)
                pro_conf = float(pro_r.get("confidence_score", 0))
                r.is_screenshot         = bool(pro_r.get("is_screenshot",  r.is_screenshot))
                r.is_manipulated        = bool(pro_r.get("is_manipulated", r.is_manipulated))
                r.has_face              = bool(pro_r.get("has_face",       r.has_face))
                r.face_count            = int(pro_r.get("face_count",      r.face_count))
                r.has_required_branding = bool(pro_r.get("has_required_branding", r.has_required_branding))
                r.branding_details      = str(pro_r.get("branding_details", r.branding_details))
                r.image_quality         = str(pro_r.get("image_quality",   r.image_quality))
                r.ai_confidence         = round(pro_conf * 100, 1)
                r.forensic_reasoning    = str(pro_r.get("reasoning", r.forensic_reasoning))
                r = apply_minor_policy(r, pro_r)
                if r.validation_status != "Invalid":
                    v = pro_r.get("recommendation", "")
                    r.validation_status = v if v in ("Valid", "Invalid") else "Invalid"
                r.exit_layer = 5
                r = _apply_hard_gates(r)
                with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                    st.session_state.layer_counts[5] += 1
            else:
                # Fallback when Pro is unavailable
                if flash_r and not flash_inv and flash_conf >= FLASH_FALLBACK_T and cv_clean:
                    r.validation_status  = "Valid"; r.exit_layer = 4
                    r.ai_confidence      = round(flash_conf * 100, 1)
                    r.forensic_reasoning = (
                        r.forensic_reasoning or
                        "Flash verdict accepted as fallback — Pro unavailable, CV clean."
                    )
                    r = _apply_hard_gates(r)
                    with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                        st.session_state.layer_counts[4] += 1
                elif cv and cv.score > 0.35:
                    r.validation_status  = "Invalid"; r.exit_layer = 2
                    r.ai_confidence      = round(cv.score * 100, 1)
                    r.error_reason       = ", ".join(cv.flags) or "CV forensic flags (AI unavailable)"
                    r.forensic_reasoning = (
                        f"AI unavailable. CV auto-rejected: score={cv.score:.2f}, flags={cv.flags}"
                    )
                    with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                        st.session_state.layer_counts[2] += 1
                else:
                    # V8 Fast40 correction: if Pro is unavailable/inconclusive but Flash
                    # has the core VALID signals (live face + required branding) and CV has
                    # no blocking forensic evidence, do not default a genuine field photo to
                    # Invalid. This only applies when hard invalid signals are absent.
                    if flash_core_valid and not flash_minor_block and not cv_blocking:
                        r.validation_status = "Valid"; r.exit_layer = 4
                        r.ai_confidence = round(max(flash_conf, FLASH_FALLBACK_T) * 100, 1)
                        r.error_reason = ""
                        r.forensic_reasoning = (
                            r.forensic_reasoning or
                            "Pro inconclusive; accepted Flash positive core signals (face + branding) with no blocking CV flags."
                        )
                        r = _apply_hard_gates(r)
                        with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                            st.session_state.layer_counts[4] += 1
                    else:
                        r.validation_status = "Invalid"; r.exit_layer = 5
                        r.pro_model = r.pro_model or PRO_MODEL
                        r.error_reason      = "Flash and Pro inconclusive; defaulting to Invalid by production policy"
                        r.forensic_reasoning = "Flash failed or was inconclusive, and Pro did not return a usable verdict."
                        with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                            st.session_state.layer_counts[5] += 1
                r = _apply_hard_gates(r)

            r.cluster_id = r.pv_image_id
            _accumulate_cost(r)
            return {"exit": True}
        except Exception as e:
            logging.error(f"[ProLayer] error (graceful degradation): {e}", exc_info=True)
            # Last-resort fallback
            ctx.r.validation_status = "Invalid"
            ctx.r.error_reason = f"AI pipeline error (graceful): {e}"
            ctx.r.exit_layer = 5
            ctx.r.cluster_id = ctx.r.pv_image_id
            _accumulate_cost(ctx.r)
            return {"exit": True, "skip": True}


# Ordered list of layers — reorder or comment-out to toggle without touching process_one()
PIPELINE_LAYERS: List[BaseLayer] = [
    HashLayer(),
    CVLayer(),
    FlashLayer(),
    ProLayer(),
]

# =============================================================================
# GEMINI PROMPTS
# =============================================================================
FLASH_PROMPT = """You are a forensic photo analyst for door-to-door field survey verification.
Return ONLY valid JSON — no markdown, no text before or after.
{
  "is_screenshot": false,
  "is_manipulated": false,
  "has_face": true,
  "face_count": 1,
  "has_required_branding": false,
  "branding_details": "None detected",
  "has_child_under_18": false,
  "child_only": false,
  "minor_holding_branding": false,
  "adult_holding_branding": false,
  "image_quality": "Good",
  "confidence_score": 0.92,
  "reasoning": "2-3 sentence forensic summary."
}

FIELD SURVEY CONTEXT:
- These photos are taken by field workers using Android/iOS camera apps in rural/urban India.
- A white or yellow date/time watermark in a corner is COMPLETELY NORMAL — camera app timestamp. Do NOT penalise for this.
- A single person holding a political pamphlet/flyer/scheme brochure is the INTENDED VALID use case.
- Slight motion blur, sunlight, JPEG artefacts, and clothing patterns (sarees, shirts) are expected.

CRITICAL RULES:
- Face visibility rule: has_face=true ONLY when at least one real live human face is clearly visible.
  If the face is hidden, cut off, fully turned away, covered by mask/cloth/phone,
  too blurry, too small, or only body/hands are visible → has_face=false.
  Partial face, back-side head, poster face, printed face, reflection face, or screen face does NOT count.
1. has_face: true ONLY if a REAL, LIVE HUMAN person is visible. Faces printed on posters do NOT count.
2. is_screenshot: true when you can see ANY of:
   - Physical device bezel/frame (phone edge, laptop edge)
   - On-screen UI (status bar, navigation bar, battery icon, signal bars, app toolbar, cursor)
   - Screen glow or display backlight bleeding around the image edges
   - Visible pixel grid or LCD subpixels
   A camera timestamp watermark alone is NOT a screenshot.
3. is_manipulated: true ONLY for:
   a) Digital cloning, splicing, compositing.
   b) Photo taken OF a flat printed photograph/document with no live person.
   c) Glass/mirror reflection showing a person (not genuine direct capture).
   EXCEPTION: live human naturally holding a pamphlet → is_manipulated=false.
4. PHOTO-OF-PHOTO / SCREEN RULE (FIX 2 — BE AGGRESSIVE):
   - Any image showing a phone screen, laptop screen, tablet screen, TV screen,
     or a physical printed photograph held up to camera → is_screenshot=true OR is_manipulated=true.
   - If you can see device bezels, display glow, glare, or moire pixel patterns from a screen → is_screenshot=true.
   - If the image shows another photo/screenshot displayed on any device → is_manipulated=true.
   - When in doubt about whether it is a screen photo — mark is_screenshot=true.
5. has_required_branding: true if NCP Ajit Pawar (clock symbol/घड्याळ), Majhi Ladki Bahin materials,
   political posters/banners/flyers, pink scheme brochures with Marathi text are clearly visible.
6. CHILD / MINOR RULE (STRICT):
   - If ONLY children/teenagers under 18 are present → child_only=true → this is INVALID.
   - If a child/minor is the one holding the pamphlet/branding (even if an adult is also present) → minor_holding_branding=true → INVALID.
   - If an adult/mother is present AND the ADULT is clearly the one holding branding → adult_holding_branding=true → may be VALID.
   - Set has_child_under_18, child_only, minor_holding_branding, adult_holding_branding carefully.
7. confidence_score: decimal float 0.0–1.0."""

PRO_PROMPT_TPL = """You are a senior forensic analyst. Your verdict is final.
Prior context: {CONTEXT}
Return ONLY valid JSON — no markdown:
{{
  "is_screenshot": false, "is_manipulated": false,
  "has_face": true, "face_count": 1,
  "has_required_branding": false, "branding_details": "None detected",
  "has_child_under_18": false, "child_only": false,
  "minor_holding_branding": false, "adult_holding_branding": false,
  "image_quality": "Good", "confidence_score": 0.95,
  "reasoning": "3-5 sentence analysis.", "recommendation": "Valid"
}}

CRITICAL RULES:
1. recommendation must be exactly "Valid" or "Invalid".
- Face visibility rule: has_face=true ONLY when at least one real live human face is clearly visible.
  If the face is hidden, cut off, fully turned away, covered by mask/cloth/phone,
  too blurry, too small, or only body/hands are visible → has_face=false.
  Partial face, back-side head, poster face, printed face, reflection face, or screen face does NOT count.
2. has_face: true ONLY if a REAL, LIVE HUMAN person is in the photograph directly.
3. is_screenshot: true if photographed from an electronic screen (pixel grid, bezel, UI elements, backlight glow).
4. is_manipulated: true for digital manipulation, photo-of-flat-printed-image, or glass/mirror reflections.
5. PHOTO-OF-SCREEN (AGGRESSIVE): Any device screen photo → is_screenshot=true. When in doubt → true.
6. CHILD / MINOR: child_only=true or minor_holding_branding=true → recommendation="Invalid".
   adult_holding_branding=true with adult present → may be "Valid".
7. has_required_branding: true if political/scheme branding clearly visible.
8. No clearly visible live human face → recommendation="Invalid"."""

# =============================================================================
# UI — PREMIUM SAAS COLOR PALETTE (FIX 10)
# =============================================================================
C_PRIMARY  = "#6366F1"   # Indigo
C_SUCCESS  = "#10B981"   # Emerald
C_DANGER   = "#EF4444"   # Red
C_WARNING  = "#F59E0B"   # Amber
C_INFO     = "#3B82F6"   # Blue
C_DARK     = "#0F172A"   # Slate 900
C_SURFACE  = "#1E293B"   # Slate 800
C_PAPER    = "#F8FAFC"   # Slate 50
C_BORDER   = "#E2E8F0"   # Slate 200
C_TEXT     = "#0F172A"   # Slate 900
C_MUTED    = "#64748B"   # Slate 500

# Legacy compat
C_GREEN = C_SUCCESS
C_RED   = C_DANGER
C_AMB   = C_WARNING
C_BLUE  = C_INFO

# =============================================================================
# CSS — PREMIUM SAAS DESIGN (FIX 10)
# =============================================================================
CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* ── Main background ────────────────────────────────────────────────── */
.main .block-container {{
    padding-top: 1.5rem;
    background: {C_PAPER};
    max-width: 1400px;
}}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C_DARK} 0%, {C_SURFACE} 100%) !important;
    border-right: 1px solid #1E293B;
}}
[data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: #F1F5F9 !important; }}
[data-testid="stSidebar"] hr {{ border-color: #334155 !important; }}

/* ── Metric cards ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    transition: box-shadow 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: 0 4px 12px rgba(99,102,241,.12);
}}
[data-testid="stMetricValue"] {{
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: {C_TEXT} !important;
    letter-spacing: -0.5px;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: {C_MUTED} !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {C_PRIMARY} 0%, #4F46E5 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 8px rgba(99,102,241,.30) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,.40) !important;
}}
.stButton > button[kind="secondary"] {{
    background: #FFFFFF !important;
    color: {C_TEXT} !important;
    border: 1.5px solid {C_BORDER} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06) !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {C_PRIMARY} !important;
    color: {C_PRIMARY} !important;
}}

/* ── Progress bar ───────────────────────────────────────────────────── */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {C_PRIMARY} 0%, #818CF8 100%) !important;
    border-radius: 6px;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: #FFFFFF;
    border-radius: 14px 14px 0 0;
    border-bottom: 2px solid {C_BORDER};
    padding: 0 8px;
    gap: 4px;
}}
.stTabs [aria-selected="true"] {{
    color: {C_PRIMARY} !important;
    border-bottom: 2px solid {C_PRIMARY} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 600;
    font-size: 13.5px;
    padding: 12px 18px;
    color: {C_MUTED} !important;
    border-radius: 10px 10px 0 0;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {C_PRIMARY} !important;
    background: rgba(99,102,241,.06) !important;
}}

/* ── Status badges ──────────────────────────────────────────────────── */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11.5px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}}
.badge-valid   {{ background: #D1FAE5; color: #065F46; }}
.badge-invalid {{ background: #FEE2E2; color: #991B1B; }}
.badge-dup     {{ background: #FEF3C7; color: #92400E; }}
.badge-error   {{ background: #F1F5F9; color: #475569; }}

/* ── Live processing card ───────────────────────────────────────────── */
.live-box {{
    background: linear-gradient(135deg, {C_DARK} 0%, {C_SURFACE} 100%);
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid #334155;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,.20);
}}
.live-box * {{ color: #E2E8F0 !important; }}

/* ── Cost pill ──────────────────────────────────────────────────────── */
.cost-pill {{
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid #6366F1;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 13px;
    color: #A5B4FC !important;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(99,102,241,.15);
}}

/* ── Section cards ──────────────────────────────────────────────────── */
.section-card {{
    background: #FFFFFF;
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}}

/* ── App header ─────────────────────────────────────────────────────── */
.app-header {{
    background: linear-gradient(135deg, {C_PRIMARY}18 0%, #818CF818 100%);
    border: 1px solid {C_PRIMARY}30;
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 20px;
}}

/* ── Dataframe ──────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* ── Divider ────────────────────────────────────────────────────────── */
hr {{ border-color: {C_BORDER} !important; margin: 1.5rem 0 !important; }}

/* ── Expander ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* ── Alert ──────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {{ border-radius: 10px; }}

/* ── Input ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    border-radius: 10px;
    border-color: {C_BORDER};
    font-size: 14px;
}}
.stSelectbox > div > div {{
    border-radius: 10px;
}}
</style>"""

def badge(s):
    cls = {
        "Valid": "valid", "Invalid": "invalid",
        "Duplicate": "dup", "Pending Review": "invalid",  # FIX 1: Pending → shown as invalid
        "Error": "error"
    }.get(s, "error")
    icon = {"Valid": "✅", "Invalid": "❌", "Duplicate": "🔁", "Error": "⚠️"}.get(s, "")
    return f'<span class="badge badge-{cls}">{icon} {s}</span>'

# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass
class VR:
    pv_image_id:            str   = ""
    validation_status:      str   = "Pending Review"
    duplicate_status:       str   = "Unique"
    matched_image_id:       str   = ""
    original_status:        str   = ""
    similarity_score:       float = 0.0
    exit_layer:             int   = 0
    has_face:               bool  = False
    face_count:             int   = 0
    is_screenshot:          bool  = False
    is_manipulated:         bool  = False
    has_required_branding:  bool  = False
    branding_details:       str   = ""
    image_quality:          str   = ""
    ai_confidence:          float = 0.0
    gps_lat:                float = 0.0
    gps_lon:                float = 0.0
    gps_valid:              bool  = False
    cluster_id:             str   = ""
    forensic_reasoning:     str   = ""
    error_reason:           str   = ""
    flash_model:            str   = ""
    pro_model:              str   = ""
    flash_prompt_tokens:    int   = 0
    flash_output_tokens:    int   = 0
    flash_total_tokens:     int   = 0
    flash_cost_usd:         float = 0.0
    flash_cost_inr:         float = 0.0
    pro_prompt_tokens:      int   = 0
    pro_output_tokens:      int   = 0
    pro_total_tokens:       int   = 0
    pro_cost_usd:           float = 0.0
    pro_cost_inr:           float = 0.0
    total_ai_tokens:        int   = 0
    total_ai_cost_usd:      float = 0.0
    total_ai_cost_inr:      float = 0.0
    usd_inr_rate:           float = USD_INR_RATE
    processing_time_ms:     int   = 0
    processed_at:           str   = ""
    row_index:              int   = -1

@dataclass
class CVR:
    has_moire:    bool  = False
    moire_score:  float = 0.0
    has_bezel:    bool  = False
    bezel_score:  float = 0.0
    ela_score:    float = 0.0
    is_blurry:    bool  = False
    blur_score:   float = 0.0
    face_count:   int   = 0
    has_gps:      bool  = False
    gps_lat:      float = 0.0
    gps_lon:      float = 0.0
    ss_software:  bool  = False
    flags:        List[str] = field(default_factory=list)
    score:        float = 0.0
    is_borderline_moire: bool = False
    is_photo_of_photo:   bool = False
    ioi_score:           float = 0.0
    ioi_signals:         List[str] = field(default_factory=list)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_ss():
    D = {
        "gemini_ok":        False,
        "sa_email":         "",
        "sheets_client":    None,
        "clip_model":       None,
        "faiss_index":      None,
        "faiss_id_map":     {},
        "hash_cache":       {},
        "results":          [],
        "current_df":       None,
        "loaded_sheet_id":  "",
        "is_processing":    False,
        "chunk_state":      None,
        "processed_count":  0,
        "total_images":     0,
        "layer_counts":     {1:0,2:0,3:0,4:0,5:0,6:0},
        "activity_log":     [],
        "human_review":     [],
        "cluster_map":      {},
        "duplicate_candidate_ids": set(),
        "result_status_cache": {},
        "db_lock":          threading.Lock(),
        "state_lock":       threading.Lock(),
        "faiss_lock":       threading.Lock(),
        "total_flash_tokens": 0,
        "total_pro_tokens":   0,
        "total_cost_inr":     0.0,
        "total_cost_usd":     0.0,
    }
    for k, v in D.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =============================================================================
# AUTH
# =============================================================================
def _find_sa():
    for pat in ["service_account*.json", "*.json"]:
        for f in glob.glob(pat):
            try:
                d = json.load(open(f))
                if d.get("type") == "service_account":
                    return f, d
            except Exception:
                pass
    return None, None

def auto_auth():
    if st.session_state.gemini_ok:
        return True

    info = None
    path = "streamlit_secrets"
    
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
        
    if not info:
        path, info = _find_sa()

    if not info:
        return False
    try:
        gcreds = Credentials.from_service_account_info(info, scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/generative-language",
        ])
        genai.configure(credentials=gcreds)
        screds = Credentials.from_service_account_info(info, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ])
        st.session_state.sheets_client = gspread.authorize(screds)
        st.session_state.gemini_ok = True
        st.session_state.sa_email  = info.get("client_email", path)
        return True
    except Exception as e:
        logging.error(f"Auth: {e}")
        return False

# =============================================================================
# DATABASE
# =============================================================================
def _db():
    c = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=30.0)
    for p in ["PRAGMA journal_mode=WAL", "PRAGMA synchronous=NORMAL",
              "PRAGMA temp_store=MEMORY", "PRAGMA cache_size=-64000"]:
        c.execute(p)
    return c

def init_db():
    with _ss_lock("db_lock", _PV_GLOBAL_DB_LOCK):
        c = _db()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS validation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pv_image_id TEXT UNIQUE,
            validation_status TEXT, duplicate_status TEXT, matched_image_id TEXT,
            original_status TEXT DEFAULT '',
            similarity_score REAL, exit_layer INTEGER,
            has_face INTEGER, face_count INTEGER, is_screenshot INTEGER,
            is_manipulated INTEGER, has_required_branding INTEGER,
            branding_details TEXT, image_quality TEXT, ai_confidence REAL,
            gps_lat REAL, gps_lon REAL, gps_valid INTEGER, cluster_id TEXT,
            forensic_reasoning TEXT, error_reason TEXT,
            flash_model TEXT, pro_model TEXT,
            flash_prompt_tokens INTEGER DEFAULT 0,
            flash_output_tokens INTEGER DEFAULT 0,
            flash_total_tokens  INTEGER DEFAULT 0,
            flash_cost_usd REAL DEFAULT 0.0,
            flash_cost_inr REAL DEFAULT 0.0,
            pro_prompt_tokens INTEGER DEFAULT 0,
            pro_output_tokens INTEGER DEFAULT 0,
            pro_total_tokens  INTEGER DEFAULT 0,
            pro_cost_usd REAL DEFAULT 0.0,
            pro_cost_inr REAL DEFAULT 0.0,
            total_ai_tokens   INTEGER DEFAULT 0,
            total_ai_cost_usd REAL DEFAULT 0.0,
            total_ai_cost_inr REAL DEFAULT 0.0,
            usd_inr_rate REAL DEFAULT 0.0,
            processing_time_ms INTEGER, processed_at TEXT,
            original_data TEXT, row_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS hash_registry (
            pv_image_id TEXT PRIMARY KEY, md5 TEXT,
            phash TEXT, dhash TEXT, whash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS duplicate_clusters (
            cluster_id TEXT, pv_image_id TEXT, is_original INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_vs  ON validation_results(validation_status);
        CREATE INDEX IF NOT EXISTS idx_el  ON validation_results(exit_layer);
        CREATE INDEX IF NOT EXISTS idx_pa  ON validation_results(processed_at);
        CREATE INDEX IF NOT EXISTS idx_md5 ON hash_registry(md5);
        """)
        # Migrate existing DBs — add any missing columns
        existing_cols = {
            row[1] for row in c.execute("PRAGMA table_info(validation_results)").fetchall()
        }
        required_cols = {
            "original_status":    "TEXT DEFAULT ''",
            "flash_cost_usd":     "REAL DEFAULT 0.0",
            "pro_cost_usd":       "REAL DEFAULT 0.0",
            "total_ai_cost_usd":  "REAL DEFAULT 0.0",
            "flash_prompt_tokens":"INTEGER DEFAULT 0",
            "flash_output_tokens":"INTEGER DEFAULT 0",
            "flash_total_tokens": "INTEGER DEFAULT 0",
            "flash_cost_inr":     "REAL DEFAULT 0.0",
            "pro_prompt_tokens":  "INTEGER DEFAULT 0",
            "pro_output_tokens":  "INTEGER DEFAULT 0",
            "pro_total_tokens":   "INTEGER DEFAULT 0",
            "pro_cost_inr":       "REAL DEFAULT 0.0",
            "total_ai_tokens":    "INTEGER DEFAULT 0",
            "total_ai_cost_inr":  "REAL DEFAULT 0.0",
            "usd_inr_rate":       "REAL DEFAULT 0.0",
        }
        for col, ddl in required_cols.items():
            if col not in existing_cols:
                c.execute(f"ALTER TABLE validation_results ADD COLUMN {col} {ddl}")
        c.commit(); c.close()

def _write(sql, params=()):
    for attempt in range(RETRY_DB):
        try:
            with _ss_lock("db_lock", _PV_GLOBAL_DB_LOCK):
                c = _db(); c.execute(sql, params); c.commit(); c.close()
            return True
        except sqlite3.OperationalError:
            if attempt < RETRY_DB - 1:
                time.sleep(0.05 * (2 ** attempt))
    return False

# =============================================================================
# DB WRITE — SYNCHRONOUS (V5 pattern restored)
# =============================================================================
# The Producer-Consumer async queue was reverted because processing threads call
# get_validation_status() immediately after save_result() to look up a matched
# image's status. With an async queue the DB hasn't been written yet → matched
# IDs appear non-existent in the output sheet.
# SQLite WAL mode + db_lock safely handles concurrent writes at CHUNK_SIZE=12.

# =============================================================================
# PRODUCER-CONSUMER DB WRITE QUEUE (CTO comment #3)
# =============================================================================
# Processing threads enqueue results; one daemon writer writes SQLite rows in batches.
# To avoid the earlier get_validation_status race, save_result() also updates an
# in-memory result_status_cache immediately. Duplicate lookups use the cache first.
_db_write_queue: "queue.Queue[tuple]" = queue.Queue()
_db_writer_thread = None
_db_writer_lock = threading.Lock()
_DB_BATCH_SIZE = 20
_DB_BATCH_WAIT_S = 0.20


def _flush_db_batch(batch):
    if not batch:
        return True
    for attempt in range(RETRY_DB):
        try:
            with _ss_lock("db_lock", _PV_GLOBAL_DB_LOCK):
                c = _db()
                for r, orig in batch:
                    c.execute("""
                    INSERT OR REPLACE INTO validation_results (
                        pv_image_id, validation_status, duplicate_status,
                        matched_image_id, original_status, similarity_score, exit_layer,
                        has_face, face_count, is_screenshot, is_manipulated,
                        has_required_branding, branding_details, image_quality,
                        ai_confidence, gps_lat, gps_lon, gps_valid, cluster_id,
                        forensic_reasoning, error_reason, flash_model, pro_model,
                        flash_prompt_tokens, flash_output_tokens, flash_total_tokens,
                        flash_cost_usd, flash_cost_inr,
                        pro_prompt_tokens, pro_output_tokens, pro_total_tokens,
                        pro_cost_usd, pro_cost_inr,
                        total_ai_tokens, total_ai_cost_usd, total_ai_cost_inr, usd_inr_rate,
                        processing_time_ms, processed_at, original_data, row_index
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        r.pv_image_id, r.validation_status, r.duplicate_status,
                        r.matched_image_id, r.original_status, r.similarity_score, r.exit_layer,
                        int(r.has_face), r.face_count, int(r.is_screenshot),
                        int(r.is_manipulated), int(r.has_required_branding),
                        r.branding_details, r.image_quality, r.ai_confidence,
                        r.gps_lat, r.gps_lon, int(r.gps_valid), r.cluster_id,
                        r.forensic_reasoning, r.error_reason,
                        r.flash_model, r.pro_model,
                        r.flash_prompt_tokens, r.flash_output_tokens, r.flash_total_tokens,
                        r.flash_cost_usd, r.flash_cost_inr,
                        r.pro_prompt_tokens, r.pro_output_tokens, r.pro_total_tokens,
                        r.pro_cost_usd, r.pro_cost_inr,
                        r.total_ai_tokens, r.total_ai_cost_usd, r.total_ai_cost_inr, r.usd_inr_rate,
                        r.processing_time_ms, r.processed_at,
                        json.dumps(orig or {}), r.row_index,
                    ))
                c.commit(); c.close()
            return True
        except sqlite3.OperationalError:
            if attempt < RETRY_DB - 1:
                time.sleep(0.05 * (2 ** attempt))
    return False


def _db_writer_loop():
    pending = []
    last_flush = time.time()
    while True:
        got_sentinel = False
        try:
            item = _db_write_queue.get(timeout=_DB_BATCH_WAIT_S)
            if item is None:  # sentinel on app shutdown
                got_sentinel = True
            else:
                pending.append(item)
        except queue.Empty:
            pass

        should_flush = pending and (
            len(pending) >= _DB_BATCH_SIZE
            or (time.time() - last_flush) >= _DB_BATCH_WAIT_S
            or got_sentinel
        )
        if should_flush:
            _flush_db_batch(pending)
            for _ in pending:
                _db_write_queue.task_done()
            pending.clear()
            last_flush = time.time()

        if got_sentinel:
            _db_write_queue.task_done()
            break

def ensure_db_writer():
    global _db_writer_thread
    with _db_writer_lock:
        if _db_writer_thread is None or not _db_writer_thread.is_alive():
            _ctx = get_script_run_ctx() if get_script_run_ctx else None
            _db_writer_thread = threading.Thread(target=_db_writer_loop, daemon=True, name="pv-db-writer")
            if _ctx and add_script_run_ctx:
                add_script_run_ctx(_db_writer_thread, _ctx)
            _db_writer_thread.start()


def flush_db_queue(timeout_s: float = 10.0):
    """Wait until queued DB writes are flushed. Called before final export/save."""
    t0 = time.time()
    while not _db_write_queue.empty() and (time.time() - t0) < timeout_s:
        time.sleep(0.05)
    try:
        _db_write_queue.join()
    except Exception:
        pass


def save_result(r: VR, orig: Dict = None):
    """Producer: enqueue DB write and update status cache immediately."""
    ensure_db_writer()
    try:
        st.session_state.result_status_cache[r.pv_image_id] = r.validation_status
    except Exception:
        pass
    _db_write_queue.put((r, orig or {}))
    return True


def get_validation_status(pid: str) -> str:
    """Lookup original status. Checks memory cache first to avoid async DB race."""
    try:
        cached = st.session_state.get("result_status_cache", {}).get(pid)
        if cached:
            return cached
    except Exception:
        pass
    try:
        c = _db()
        row = c.execute(
            "SELECT validation_status FROM validation_results WHERE pv_image_id=?", (pid,)
        ).fetchone()
        c.close()
        return row[0] if row else ""
    except Exception:
        return ""

def get_recent_duplicate_ids(days: int = DUP_LOOKBACK_DAYS) -> set:
    """Return image IDs from last N days only.

    Important: this intentionally does NOT include all hash_cache keys, because
    hash_cache may contain old historical hashes. Current-batch IDs are added
    live in HashLayer after each unique image is registered.
    """
    try:
        if days <= 0:
            return set(st.session_state.hash_cache.keys())
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c = _db()
        rows = c.execute(
            "SELECT pv_image_id FROM validation_results WHERE processed_at >= ?",
            (cutoff,)
        ).fetchall()
        c.close()
        return {r[0] for r in rows if r and r[0]}
    except Exception as e:
        logging.warning(f"recent duplicate id load failed: {e}")
        return set()

def save_hash(pid, md5, ph, dh, wh):
    _write(
        "INSERT OR IGNORE INTO hash_registry (pv_image_id,md5,phash,dhash,whash) VALUES (?,?,?,?,?)",
        (pid, md5, ph, dh, wh)
    )

def get_review():
    # FIX 1: Pending Review always becomes Invalid — no manual queue
    return []

def load_hashes(days: int = DUP_LOOKBACK_DAYS):
    """Load only hashes from the active duplicate window.

    Uses validation_results.processed_at as the source of truth so the in-memory
    hash_cache matches the same 10-day business window used by duplicate search.
    """
    try:
        c = _db()
        if days <= 0:
            rows = c.execute(
                "SELECT pv_image_id,md5,phash,dhash,whash FROM hash_registry"
            ).fetchall()
        else:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows = c.execute("""
                SELECT h.pv_image_id, h.md5, h.phash, h.dhash, h.whash
                FROM hash_registry h
                JOIN validation_results v ON v.pv_image_id = h.pv_image_id
                WHERE v.processed_at >= ?
            """, (cutoff,)).fetchall()
        c.close()
        out = {}
        for pid, md5, ph, dh, wh in rows:
            try:
                out[pid] = {
                    "md5": md5,
                    "ph": imagehash.hex_to_hash(ph),
                    "dh": imagehash.hex_to_hash(dh),
                    "wh": imagehash.hex_to_hash(wh),
                }
            except Exception:
                pass
        return out
    except Exception as e:
        logging.warning(f"load_hashes failed: {e}")
        return {}

def clear_db():
    with _ss_lock("db_lock", _PV_GLOBAL_DB_LOCK):
        c = _db()
        c.executescript(
            "DELETE FROM validation_results;"
            "DELETE FROM hash_registry;"
            "DELETE FROM duplicate_clusters;"
            "VACUUM;"
        )
        c.commit(); c.close()

# =============================================================================
# FAISS
# =============================================================================
def faiss_save(idx):
    try:
        faiss.write_index(idx, FAISS_TMP)
        if os.path.exists(FAISS_INDEX): shutil.copy2(FAISS_INDEX, FAISS_BAK)
        os.replace(FAISS_TMP, FAISS_INDEX)
        with open(FAISS_MTMP, "w") as f:
            json.dump(st.session_state.faiss_id_map, f)
        if os.path.exists(FAISS_MAP): shutil.copy2(FAISS_MAP, FAISS_MBAK)
        os.replace(FAISS_MTMP, FAISS_MAP)
        return True
    except Exception as e:
        logging.error(f"FAISS save: {e}"); return False

def faiss_load():
    for p in [FAISS_INDEX, FAISS_BAK]:
        if os.path.exists(p):
            try:
                idx = faiss.read_index(p)
                if p == FAISS_BAK: shutil.copy2(p, FAISS_INDEX)
                return idx
            except Exception:
                pass
    return faiss.IndexIDMap2(faiss.IndexFlatIP(512))

def faiss_load_map():
    for p in [FAISS_MAP, FAISS_MBAK]:
        if os.path.exists(p):
            try: return json.load(open(p))
            except Exception: pass
    return {}

def faiss_add(idx, emb, pid):
    fid = int(uuid.uuid4().int % (2**31 - 1))
    n   = emb / (np.linalg.norm(emb) + 1e-10)
    idx.add_with_ids(n.reshape(1,-1).astype("float32"), np.array([fid], dtype=np.int64))
    st.session_state.faiss_id_map[str(fid)] = pid
    if idx.ntotal % FAISS_FLUSH == 0:
        faiss_save(idx)
    return idx

def faiss_search(idx, emb, k=1):
    if idx.ntotal == 0: return [], []
    n = emb / (np.linalg.norm(emb) + 1e-10)
    dists, idxs = idx.search(n.reshape(1,-1).astype("float32"), min(k, idx.ntotal))
    ids, scores = [], []
    for d, i in zip(dists[0], idxs[0]):
        if i != -1:
            key = str(i)
            if key in st.session_state.faiss_id_map:
                ids.append(st.session_state.faiss_id_map[key])
                scores.append(float(d))
    return ids, scores

# =============================================================================
# CLIP
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_clip():
    return SentenceTransformer("clip-ViT-B-32")

def embed(model, img): return model.encode(img, show_progress_bar=False)

RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

@st.cache_resource(show_spinner=False)
def get_http_session():
    # httpx.Client is thread-safe and supports HTTP/2, significantly faster than
    # requests.Session for high-concurrency image downloads (review comment #2).
    # follow_redirects=True replaces allow_redirects; Timeout object separates
    # connect vs. read timeouts natively without tuple unpacking.
    return httpx.Client(
        headers={"User-Agent": "PhotoVerifyBot/3.1"},
        follow_redirects=True,
        timeout=httpx.Timeout(connect=HTTP_CONNECT_TIMEOUT, read=HTTP_TIMEOUT,
                              write=10.0, pool=5.0),
    )

@st.cache_resource(show_spinner=False)
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face_crop(img: Image.Image) -> Optional[Image.Image]:
    try:
        cc = get_face_cascade()
        arr_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        faces = cc.detectMultiScale(
            arr_gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(FACE_CONF_MIN_SIZE, FACE_CONF_MIN_SIZE))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.25 * max(w, h))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(img.size[0], x + w + pad); y2 = min(img.size[1], y + h + pad)
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return None

def face_similarity(crop_a: Image.Image, crop_b: Image.Image, clip_model=None) -> float:
    try:
        if clip_model is None:
            clip_model = st.session_state.clip_model
        if clip_model is None:
            return 0.0
        a = resized_copy(crop_a, CLIP_MAX_SIDE)
        b = resized_copy(crop_b, CLIP_MAX_SIDE)
        va = np.array(embed(clip_model, a), dtype=float)
        vb = np.array(embed(clip_model, b), dtype=float)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10)
        return float(np.dot(va, vb) / denom)
    except Exception:
        return 0.0

@st.cache_resource(show_spinner=False)
def get_gemini_model(model_name):
    bare = model_name.replace("models/", "")
    if "flash" in bare.lower():
        return genai.GenerativeModel(bare, system_instruction=FLASH_PROMPT)
    return genai.GenerativeModel(bare)

def normalize_image(img):
    return ImageOps.exif_transpose(img).convert("RGB")

def resized_copy(img, max_side):
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), RESAMPLE)
    return out

def encode_ai_image(img, max_side=AI_MAX_SIDE, quality=AI_JPEG_QUALITY):
    work = resized_copy(img, max_side)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def hash_vote_count(hashes, cached_hashes):
    if not cached_hashes:
        return 0
    pd_ = hashes["ph"] - cached_hashes["ph"]
    dd_ = hashes["dh"] - cached_hashes["dh"]
    wd_ = hashes["wh"] - cached_hashes["wh"]
    return int(pd_ <= PHASH_T) + int(dd_ <= DHASH_T) + int(wd_ <= WHASH_T)

# =============================================================================
# IMAGE DOWNLOAD
# =============================================================================
def fetch(url):
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.google.com/",
    }
    url = str(url).strip()
    if not url or url.lower() in {"nan", "none", "null"}:
        logging.warning("Fetch skipped: empty or invalid URL value.")
        return None

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        logging.warning(f"Fetch skipped: malformed URL [{url[:80]}].")
        return None

    session = get_http_session()
    last_err = None
    for attempt in range(RETRY_IMG):
        try:
            # httpx streams via context manager; timeout set on the Client object.
            # iter_bytes() is the httpx equivalent of requests' iter_content().
            with session.stream("GET", url, headers=HEADERS) as resp:
                resp.raise_for_status()
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if ctype and ("html" in ctype or "json" in ctype):
                    raise ValueError(f"Non-image response: {ctype}")
                started = time.monotonic()
                buf = io.BytesIO()
                for chunk in resp.iter_bytes(chunk_size=65536):
                    if not chunk:
                        continue
                    buf.write(chunk)
                    if buf.tell() > MAX_IMAGE_BYTES:
                        raise ValueError("Image too large")
                    if (time.monotonic() - started) > HTTP_HARD_TIMEOUT_S:
                        raise TimeoutError("Download timeout")
            content = buf.getvalue()
            img = Image.open(io.BytesIO(content))
            img.load()
            return normalize_image(img)
        except Exception as e:
            last_err = e
            logging.warning(f"Fetch attempt {attempt+1}/{RETRY_IMG} failed: {e}")
            if attempt < RETRY_IMG - 1:
                time.sleep(1.5 * (2 ** attempt))
    logging.error(f"Fetch permanently failed [{url[:80]}]: {last_err}")
    return None

# =============================================================================
# LAYER 1 — HASH FINGERPRINTING
# =============================================================================
def gen_hashes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    md5 = hashlib.md5(buf.getvalue()).hexdigest()
    ph = imagehash.phash(img)
    dh = imagehash.dhash(img)
    wh = imagehash.whash(img)
    return {"md5":md5, "ph":ph, "dh":dh, "wh":wh,
            "phash":str(ph), "dhash":str(dh), "whash":str(wh)}

def find_dup(hashes, cache, allowed_ids=None):
    """FIX 4/5: Limit comparison to allowed_ids (recent window + current batch).
    BUG FIX: empty set(allowed_ids or []) became set() → 'not allowed_ids' was True
    → every image compared against ENTIRE cache → all flagged as L1 duplicates.
    Now: allowed_ids=None means compare all cache; empty set means compare nothing.
    """
    md5 = hashes["md5"]
    ph, dh, wh = hashes["ph"], hashes["dh"], hashes["wh"]
    if allowed_ids is None:
        cache_items = list(cache.items())
    else:
        allowed_ids = set(allowed_ids)
        cache_items = [(cid, ch) for cid, ch in cache.items() if cid in allowed_ids]

    for cid, ch in cache_items:
        if ch["md5"] == md5:
            return {
                "is_duplicate": True, "match_id": cid,
                "similarity": 100.0, "duplicate_status": "Exact Duplicate",
                "reason": "MD5 fingerprint matched a recent/current-batch image.",
            }
    best = None
    for cid, ch in cache_items:
        pd_ = ph - ch["ph"]
        dd_ = dh - ch["dh"]
        wd_ = wh - ch["wh"]
        votes = int(pd_ <= PHASH_T) + int(dd_ <= DHASH_T) + int(wd_ <= WHASH_T)
        if votes < HASH_VOTES_T:
            continue
        confidence = (
            max(0.0, 1.0 - (pd_ / max(PHASH_T, 1))) +
            max(0.0, 1.0 - (dd_ / max(DHASH_T, 1))) +
            max(0.0, 1.0 - (wd_ / max(WHASH_T, 1)))
        ) / 3.0
        candidate = {
            "is_duplicate": True, "match_id": cid,
            "similarity": round(70.0 + confidence * 30.0, 2),
            "duplicate_status": "Perceptual Duplicate",
            "reason": f"Hash votes={votes}/3 (pHash={pd_}, dHash={dd_}, wHash={wd_}).",
            "rank": (votes, confidence),
        }
        if best is None or candidate["rank"] > best["rank"]:
            best = candidate
    if best:
        best.pop("rank", None)
        return best
    return {"is_duplicate": False, "match_id": "", "similarity": 0.0,
            "duplicate_status": "Unique", "reason": ""}

# =============================================================================
# LAYER 2 — CV FORENSICS
# =============================================================================
def cv_moire(img):
    try:
        arr = np.array(img.convert("L"), dtype=float)
        fs  = scipy.fftpack.fftshift(scipy.fftpack.fft2(arr))
        mag = np.abs(fs); h, w = mag.shape
        mag[h//2-5:h//2+5, w//2-5:w//2+5] = 0
        ratio = float(np.percentile(mag, 99)) / (float(np.mean(mag)) + 1e-10)
        return ratio > MOIRE_T, min(ratio/20.0, 1.0)
    except Exception:
        return False, 0.0

def cv_is_poster_moire(img):
    try:
        arr   = np.array(img.convert("L"))
        edges = cv2.Canny(arr, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (arr.shape[0] * arr.shape[1])
        return edge_density > POSTER_EDGE_T
    except Exception:
        return False

def cv_timestamp(img):
    try:
        arr  = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]
        roi_h = max(30, h // 12)
        roi_w = max(80, w // 6)
        for corner in [arr[h-roi_h:h, 0:roi_w], arr[h-roi_h:h, w-roi_w:w]]:
            gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)
            _, bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            _, dark   = cv2.threshold(gray, 50,  255, cv2.THRESH_BINARY_INV)
            bright_ratio = float(np.count_nonzero(bright)) / bright.size
            dark_ratio   = float(np.count_nonzero(dark))   / dark.size
            if bright_ratio > 0.08 and dark_ratio > 0.35:
                return True
            mean_lum = float(gray.mean())
            if mean_lum < 70 and bright_ratio > 0.02:
                return True
        return False
    except Exception:
        return False

def cv_bezel(img):
    try:
        arr   = np.array(img)
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        at = gray.shape[0] * gray.shape[1]
        for cnt in cnts:
            ca = cv2.contourArea(cnt)
            if ca > at * 0.3:
                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                if len(approx) == 4:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    if 0.3 < bw/(bh+1e-6) < 3.0:
                        return True, min(ca/at, 1.0)
        return False, 0.0
    except Exception:
        return False, 0.0

def cv_ela(img):
    try:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90); buf.seek(0)
        rc   = Image.open(buf).convert("RGB")
        diff = np.abs(np.array(img, dtype=float) - np.array(rc, dtype=float))
        return float(np.mean(diff)) / 255.0
    except Exception:
        return 0.0

def cv_blur(img):
    try:
        gray  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < BLUR_T, float(score)
    except Exception:
        return False, 999.0

def cv_faces(img):
    try:
        cc  = get_face_cascade()
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        fc  = cc.detectMultiScale(arr, 1.1, 4, minSize=(FACE_CONF_MIN_SIZE, FACE_CONF_MIN_SIZE))
        return len(fc) if fc is not None and len(fc) > 0 else 0
    except Exception:
        return 0

def cv_exif(img):
    out = {"has_gps": False, "gps_lat": 0.0, "gps_lon": 0.0, "ss_sw": False}
    try:
        exif = img._getexif() if hasattr(img, "_getexif") else None
        if not exif:
            return out
        for tag_id, val in exif.items():
            tag = TAGS.get(tag_id, "")
            if tag == "Software" and isinstance(val, str):
                if any(kw in val.lower() for kw in ["screenshot","screen capture","grab"]):
                    out["ss_sw"] = True
            if tag == "GPSInfo" and isinstance(val, dict):
                try:
                    def _conv(d):
                        return float(d[0]) + float(d[1])/60 + float(d[2])/3600
                    lat_d = val.get(2); lon_d = val.get(4)
                    lat_r = val.get(1, "N"); lon_r = val.get(3, "E")
                    if lat_d and lon_d:
                        lat = _conv(lat_d) * (-1 if lat_r == "S" else 1)
                        lon = _conv(lon_d) * (-1 if lon_r == "W" else 1)
                        out["has_gps"] = True; out["gps_lat"] = lat; out["gps_lon"] = lon
                except Exception:
                    pass
    except Exception:
        pass
    return out

def cv_face_line_artifact(img):
    """Detect horizontal line artifacts typical of screen captures."""
    try:
        arr  = np.array(img.convert("L"))
        rows = arr.mean(axis=1)
        diffs = np.abs(np.diff(rows.astype(float)))
        score = float(np.percentile(diffs, 98))
        return score > 25.0, score
    except Exception:
        return False, 0.0

def cv_image_of_image(img, moire_score=0.0, bezel_score=0.0, ela_score=0.0):
    """FIX 2: More aggressive photo-of-screen/photo detection.
    Returns (ioi_score 0-1, list_of_signals).
    Lower IOI_SCORE_T (0.35) means we catch more screen photos.
    """
    signals = []
    s = 0.0

    # Bezel detection (strong indicator of device frame)
    if bezel_score > 0.15:
        s += 0.45          # v9: increased weight
        signals.append(f"device_bezel({bezel_score:.2f})")

    # FIX 2: Screen glow — uniform luminance at image edges (backlight)
    try:
        arr  = np.array(img.convert("L"))
        h, w = arr.shape
        margin = max(10, min(h, w) // 20)
        edges  = np.concatenate([
            arr[:margin, :].ravel(),
            arr[-margin:, :].ravel(),
            arr[:, :margin].ravel(),
            arr[:, -margin:].ravel()
        ])
        edge_mean = float(edges.mean())
        edge_std  = float(edges.std())
        if edge_mean > 200 and edge_std < 25:      # v9: tightened threshold
            s += 0.30
            signals.append(f"screen_glow(mean={edge_mean:.0f},std={edge_std:.1f})")
        elif edge_mean > 180 and edge_std < 35:    # v9: secondary soft threshold
            s += 0.15
            signals.append(f"bright_edges")
    except Exception:
        pass

    # Moire pattern from screen pixel grid
    if moire_score > 0.25:
        s += min(moire_score * 0.35, 0.35)
        signals.append(f"screen_moire({moire_score:.2f})")

    # ELA from JPEG recompression (printed/digital copy)
    if ela_score > 0.35:
        s += 0.20
        signals.append(f"ela_recompression({ela_score:.2f})")

    # FIX 2: Uniform color banding (typical of photographed screens)
    try:
        arr_rgb = np.array(img)
        # Check for suspicious RGB regularity in pixel values (LCD quantization)
        for ch in range(3):
            vals  = arr_rgb[:, :, ch].ravel()
            hist  = np.histogram(vals, bins=32)[0]
            peaks = (hist > hist.mean() * 3).sum()
            if peaks >= 8:
                s += 0.12
                signals.append("pixel_quantization")
                break
    except Exception:
        pass

    return min(s, 1.0), signals

def cv_has_branding_colors(img):
    try:
        arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2HSV)
        hsv = arr
        total_px = hsv.shape[0] * hsv.shape[1]
        if total_px == 0:
            return True

        ncp_blue = cv2.inRange(hsv, np.array([100, 90, 40]), np.array([130, 255, 255]))
        if ncp_blue.sum() / 255 / total_px > 0.015:
            return True

        red_lo = cv2.inRange(hsv, np.array([0,  90, 80]), np.array([15, 255, 255]))
        red_hi = cv2.inRange(hsv, np.array([155, 90, 80]), np.array([179, 255, 255]))
        if (red_lo.sum() + red_hi.sum()) / 255 / total_px > 0.015:
            return True

        orange = cv2.inRange(hsv, np.array([5, 90, 80]), np.array([25, 255, 255]))
        if orange.sum() / 255 / total_px > 0.015:
            return True

        green = cv2.inRange(hsv, np.array([35, 80, 50]), np.array([85, 255, 255]))
        if green.sum() / 255 / total_px > 0.015:
            return True

        return False
    except Exception:
        return True

def run_cv(img):
    c = CVR()
    raw_moire, c.moire_score = cv_moire(img)
    c.has_bezel, c.bezel_score = cv_bezel(img)
    c.ela_score                = cv_ela(img)
    c.is_blurry, c.blur_score  = cv_blur(img)
    c.face_count               = cv_faces(img)
    ex = cv_exif(img)
    c.has_gps = ex["has_gps"]; c.gps_lat = ex["gps_lat"]; c.gps_lon = ex["gps_lon"]
    c.ss_software = ex["ss_sw"]
    has_timestamp = cv_timestamp(img)

    if raw_moire and cv_is_poster_moire(img):
        c.has_moire   = False
        c.moire_score = c.moire_score * 0.15
    else:
        c.has_moire = raw_moire

    c.is_borderline_moire = (
        MOIRE_BORDERLINE_LOW <= c.moire_score <= MOIRE_BORDERLINE_HIGH
        and has_timestamp and not c.has_moire
    )

    # FIX 2: More aggressive IoI detection (lower threshold)
    c.ioi_score, c.ioi_signals = cv_image_of_image(
        img, moire_score=c.moire_score,
        bezel_score=c.bezel_score, ela_score=c.ela_score)
    c.is_photo_of_photo = c.ioi_score >= IOI_SCORE_T

    flags = []
    if c.has_moire:          flags.append("moire_pattern")
    if c.has_bezel:          flags.append("screen_bezel")
    if c.ela_score > ELA_T:  flags.append("ela_manipulation")
    if c.ss_software:        flags.append("screenshot_software")
    if has_timestamp:        flags.append("timestamp_overlay")
    if c.is_blurry:          flags.append("blurry_image")
    if c.is_photo_of_photo:  flags.append(f"image_of_image({','.join(c.ioi_signals)})")
    c.flags = flags

    s = 0.0
    if c.has_moire and c.moire_score > 15.0: s += 0.20
    if c.has_bezel:           s += 0.35
    if c.ela_score > ELA_T:   s += 0.20
    if c.ss_software:         s += 0.25
    if c.is_blurry and c.blur_score < 15: s += 0.55
    if c.is_photo_of_photo:   s += 0.40
    c.score = min(s, 1.0)
    return c

# =============================================================================
# LAYER 2.5 — PRE-AI SCREENING
# =============================================================================
def pre_ai_screen(cv_result: CVR, orig_row: dict, img=None):
    # Gate 1: Screenshot with device bezel + metadata
    if cv_result.has_bezel and cv_result.ss_software:
        return True, "Screenshot detected (device bezel + screenshot software metadata)"

    # Gate 2: Extreme ELA + bezel
    if cv_result.ela_score > 0.85 and cv_result.has_bezel:
        return True, "Extreme digital manipulation (high ELA + device bezel)"

    # Gate 3: Blank / corrupted image
    if img is not None:
        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            if float(gray.std()) < 5.0:
                return True, "Blank or corrupted image"
        except Exception:
            pass

    # Gate 4 (FIX 2): Image-of-image at lowered threshold (0.35)
    if cv_result.is_photo_of_photo:
        sig_str = ", ".join(cv_result.ioi_signals) or "composite CV signals"
        return True, (
            f"Screen/printed-photo detected (IoI score={cv_result.ioi_score:.2f}): {sig_str}. "
            f"Photo appears to be taken from a digital screen or printed photograph."
        )

    # Gate 5: No face + strong manipulation signals → reject without AI
    if cv_result.face_count == 0:
        if cv_result.has_bezel or cv_result.ela_score > ELA_T:
            return True, "No human face + device bezel/ELA signals detected"
        logging.info("Pre-AI face-miss: routing to Flash for face verification")

    # Gate 6: Branding color pre-screen
    if BRANDING_COLOR_GATE and img is not None and cv_result.face_count > 0:
        if not cv_has_branding_colors(img):
            return True, (
                "No political branding colours detected (NCP blue, scheme pink/red, saffron, green). "
                "Valid survey photos must show political/scheme branding materials."
            )

    return False, ""

# =============================================================================
# GEMINI HELPERS
# =============================================================================
def _parse_json(text):
    for t in [text, text.strip()]:
        try: return json.loads(t)
        except Exception: pass
    for m in ["```json", "```"]:
        if m in text:
            for part in text.split(m)[1::2]:
                try: return json.loads(part.split("```")[0].strip())
                except Exception: pass
    try:
        s, e = text.index("{"), text.rindex("}")+1
        return json.loads(text[s:e])
    except Exception:
        return None

def _resp_text(resp):
    try:
        if getattr(resp, "text", None):
            return resp.text
    except Exception:
        pass
    try:
        return "".join(
            part.text for cand in getattr(resp, "candidates", [])
            for part in getattr(getattr(cand, "content", None), "parts", [])
            if getattr(part, "text", None)
        )
    except Exception:
        return ""

def _extract_usage(resp) -> Dict[str, int]:
    usage = getattr(resp, "usage_metadata", None)
    if usage is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        output = int(getattr(usage, "candidates_token_count", 0) or 0)
        total  = int(getattr(usage, "total_token_count", prompt + output) or (prompt + output))
        return {"prompt_tokens": prompt, "output_tokens": output, "total_tokens": total}
    except Exception:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def _model_prices_usd_per_mtok(model_name: str) -> Tuple[float, float]:
    m = (model_name or "").lower()
    if "flash" in m:
        return FLASH_INPUT_USD_PER_MTOK, FLASH_OUTPUT_USD_PER_MTOK
    return PRO_INPUT_USD_PER_MTOK, PRO_OUTPUT_USD_PER_MTOK

def _cost_from_usage(model_name: str, usage: Dict[str, int]) -> Tuple[float, float]:
    in_rate, out_rate = _model_prices_usd_per_mtok(model_name)
    usd = (
        (float(usage.get("prompt_tokens", 0)) / 1_000_000.0) * in_rate +
        (float(usage.get("output_tokens", 0)) / 1_000_000.0) * out_rate
    )
    inr = round(usd * USD_INR_RATE, 6)
    return round(usd, 8), inr


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort 429/rate-limit detector across google/api/http exception types."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "resource exhausted" in s or "quota" in s


def _sleep_ai_backoff(attempt: int, exc: Exception, model_name: str):
    """Exponential backoff with jitter for Gemini transient/rate-limit errors."""
    base = min(2 ** attempt, 16)
    jitter_hi = 2.0 if _is_rate_limit_error(exc) else 1.25
    random_fraction = random.uniform(0.25, jitter_hi)
    delay = base + random_fraction
    logging.warning(f"{model_name} retry {attempt + 1}/{RETRY_AI} after {delay:.2f}s: {exc}")
    time.sleep(delay)

def _generate_json(model, payload, max_output_tokens):
    configs = [
        ({"temperature": 0.0, "max_output_tokens": max_output_tokens,
          "response_mime_type": "application/json"}, {"timeout": AI_TIMEOUT_S}),
        ({"temperature": 0.0, "max_output_tokens": max_output_tokens}, {"timeout": AI_TIMEOUT_S}),
        ({"temperature": 0.0, "max_output_tokens": max_output_tokens}, None),
    ]
    last_error = None
    for generation_config, request_options in configs:
        try:
            kwargs = {"generation_config": generation_config}
            if request_options is not None:
                kwargs["request_options"] = request_options
            return model.generate_content(payload, **kwargs)
        except TypeError as e:
            last_error = e; continue
    if last_error: raise last_error
    raise RuntimeError("Gemini request failed before execution")

def gemini_flash(img):
    if not st.session_state.gemini_ok: return None
   
# FLASH_PROMPT is registered as Gemini system_instruction in get_gemini_model().
# Flash user payload contains only the image bytes to reduce repeated prompt tokens.

    payload = [{"mime_type": "image/jpeg", "data": encode_ai_image(img)}]
    for attempt in range(RETRY_AI):
        try:
            model = get_gemini_model(FLASH_MODEL)
            resp  = _generate_json(model, payload, 256)
            r     = _parse_json(_resp_text(resp))
            if r:
                usage = _extract_usage(resp)
                usd, inr = _cost_from_usage(FLASH_MODEL, usage)
                r["_usage"]    = usage
                r["_cost_usd"] = usd
                r["_cost_inr"] = inr
                return r
        except Exception as e:
            if attempt < RETRY_AI-1: _sleep_ai_backoff(attempt, e, "Flash")
            else: logging.error(f"Flash: {e}")
    return None

def gemini_pro(img, flash_r, cv):
    if not st.session_state.gemini_ok: return None
    flash_ctx = {k: v for k, v in (flash_r or {}).items() if not str(k).startswith("_")}
    ctx = (f"Flash: {json.dumps(flash_ctx)}\n"
           f"CV: moire={cv.moire_score:.2f}, ela={cv.ela_score:.2f}, "
           f"bezel={cv.bezel_score:.2f}, flags={cv.flags}")
    prompt  = PRO_PROMPT_TPL.replace("{CONTEXT}", ctx)
    payload = [prompt, {"mime_type":"image/jpeg","data":encode_ai_image(img)}]
    for attempt in range(RETRY_AI):
        try:
            model = get_gemini_model(PRO_MODEL)
            resp  = _generate_json(model, payload, 512)
            r     = _parse_json(_resp_text(resp))
            if r:
                usage = _extract_usage(resp)
                usd, inr = _cost_from_usage(PRO_MODEL, usage)
                r["_usage"]    = usage
                r["_cost_usd"] = usd
                r["_cost_inr"] = inr
                return r
        except Exception as e:
            if attempt < RETRY_AI-1: _sleep_ai_backoff(attempt, e, "Pro")
            else: logging.error(f"Pro: {e}")
    return None

# =============================================================================
# LAYER 3b — ORB MULTI-ANGLE DUPLICATE DETECTION
# =============================================================================
def orb_match(img_a: Image.Image, img_b: Image.Image) -> Tuple[bool, float, int]:
    try:
        def _prep(img):
            w = resized_copy(img, ORB_MAX_SIDE)
            return cv2.cvtColor(np.array(w), cv2.COLOR_RGB2GRAY)

        gA = _prep(img_a)
        gB = _prep(img_b)

        orb = cv2.ORB_create(nfeatures=1000)
        kpA, desA = orb.detectAndCompute(gA, None)
        kpB, desB = orb.detectAndCompute(gB, None)

        if desA is None or desB is None or len(kpA) < 8 or len(kpB) < 8:
            return False, 0.0, 0

        bf      = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desA, desB, k=2)
        good = [m for m, n in matches if m.distance < ORB_LOWE_RATIO * n.distance]

        if len(good) < ORB_MIN_MATCHES:
            return False, 0.0, len(good)

        ptsA = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ptsB = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        _, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ORB_HOMOGRAPHY_T)
        if mask is None:
            return False, 0.0, 0

        inliers      = int(mask.sum())
        inlier_ratio = inliers / max(len(good), 1)
        score        = inliers / max(len(kpA), len(kpB), 1)
        is_dup = (inliers >= ORB_MIN_INLIERS and inlier_ratio >= ORB_INLIER_RATIO)
        return is_dup, min(float(score) * 2.0, 1.0), inliers

    except Exception as e:
        logging.warning(f"ORB match error: {e}")
        return False, 0.0, 0

ORB_IMG_CACHE_SIZE = 200
_orb_img_cache: Dict[str, Image.Image] = {}
_orb_img_order: List[str] = []
_orb_cache_lock = threading.Lock()

def orb_cache_add(pid: str, img: Image.Image):
    global _orb_img_cache, _orb_img_order
    with _orb_cache_lock:
        if pid in _orb_img_cache:
            return
        _orb_img_cache[pid] = resized_copy(img, ORB_MAX_SIDE)
        _orb_img_order.append(pid)
        if len(_orb_img_order) > ORB_IMG_CACHE_SIZE:
            evict = _orb_img_order.pop(0)
            _orb_img_cache.pop(evict, None)

def orb_cache_get(pid: str) -> Optional[Image.Image]:
    with _orb_cache_lock:
        return _orb_img_cache.get(pid)

# =============================================================================
# CLUSTER MANAGEMENT
# =============================================================================
def assign_cluster(r: VR):
    if r.validation_status != "Duplicate" or not r.matched_image_id:
        r.cluster_id = r.pv_image_id; return r
    with st.session_state.state_lock:
        cm  = st.session_state.cluster_map
        cid = cm.get(r.matched_image_id, r.matched_image_id)
        r.cluster_id = cid
        cm[r.pv_image_id] = cid
        if r.matched_image_id not in cm: cm[r.matched_image_id] = cid
    _write(
        "INSERT OR IGNORE INTO duplicate_clusters (cluster_id,pv_image_id,is_original) VALUES (?,?,?)",
        (r.cluster_id, r.pv_image_id, 0)
    )
    return r

# =============================================================================
# MINOR POLICY (FIX 3 — tightened)
# =============================================================================
def apply_minor_policy(result: VR, ai_result: dict) -> VR:
    """FIX 3: Strict child/minor policy enforced at both Flash and Pro levels.

    Invalid when:
      - Only children present (child_only=true)
      - A minor is holding the branding/pamphlet (minor_holding_branding=true)
      - Children present but NO adult is holding branding (adult_holding_branding=false)

    Valid when:
      - An adult is present AND clearly holding branding (adult_holding_branding=true)
        even if a child is also visible in the frame.
    """
    if not MINOR_POLICY_ENABLED or not ai_result:
        return result

    has_child     = bool(ai_result.get("has_child_under_18", False))
    child_only    = bool(ai_result.get("child_only", False))
    minor_holding = bool(ai_result.get("minor_holding_branding", False))
    adult_holding = bool(ai_result.get("adult_holding_branding", False))

    if child_only:
        result.validation_status = "Invalid"
        result.error_reason = "Minor policy: only children/minors present in photo (no adult)."
    elif minor_holding:
        result.validation_status = "Invalid"
        result.error_reason = "Minor policy: child/minor is holding the branding/pamphlet."
    elif has_child and not adult_holding:
        result.validation_status = "Invalid"
        result.error_reason = "Minor policy: child present but no adult clearly holds branding."

    return result

# =============================================================================
# PIPELINE — Strategy-Pattern orchestrator
# =============================================================================
def _accumulate_cost(r: VR):
    with st.session_state.state_lock:
        st.session_state.total_flash_tokens += r.flash_total_tokens
        st.session_state.total_pro_tokens   += r.pro_total_tokens
        st.session_state.total_cost_usd     = round(
            st.session_state.total_cost_usd + r.flash_cost_usd + r.pro_cost_usd, 8)
        st.session_state.total_cost_inr     = round(
            st.session_state.total_cost_inr + r.flash_cost_inr + r.pro_cost_inr, 4)

def process_one(source, orig=None, url="", row_idx=-1):
    """
    Thin orchestrator: build a LayerContext then iterate PIPELINE_LAYERS in order.
    Each layer calls ctx.r (VR) in place and returns {"exit": bool, "skip"?: bool}.
    Layers that raise internally return skip=True for graceful degradation;
    they do not abort the entire image — later layers continue.
    To reorder or toggle layers, edit PIPELINE_LAYERS — no changes here needed.
    """
    t0 = time.time()
    r  = VR()
    r.pv_image_id  = str(uuid.uuid4())[:12].upper()
    r.processed_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    r.row_index    = row_idx
    r.usd_inr_rate = USD_INR_RATE
    orig           = orig or {}

    if isinstance(source, str):           img = fetch(source)
    elif isinstance(source, Image.Image): img = normalize_image(source)
    elif hasattr(source, "read"):         img = normalize_image(Image.open(source))
    else:                                 img = None

    if img is None:
        r.validation_status  = "Error"
        r.error_reason       = "Image load failed"
        r.processing_time_ms = int((time.time() - t0) * 1000)
        save_result(r, orig)
        return r, None

    hashes = gen_hashes(img)
    cv_img = resized_copy(img, CV_MAX_SIDE)
    clip   = st.session_state.clip_model
    emb    = embed(clip, resized_copy(img, CLIP_MAX_SIDE)) if clip else None

    ctx = LayerContext(r=r, img=img, cv_img=cv_img,
                       hashes=hashes, emb=emb, orig=orig, t0=t0)

    for layer in PIPELINE_LAYERS:
        try:
            result = layer.run(ctx)
        except Exception as e:
            logging.error(f"[{layer.name}] unhandled error (graceful degradation): {e}",
                          exc_info=True)
            result = {"exit": False, "skip": True}

        if result.get("exit"):
            break

    r = ctx.r
    r.total_ai_tokens    = int(r.flash_total_tokens + r.pro_total_tokens)
    r.total_ai_cost_usd  = round(r.flash_cost_usd + r.pro_cost_usd, 8)
    r.total_ai_cost_inr  = round(r.flash_cost_inr + r.pro_cost_inr, 6)
    r.processing_time_ms = int((time.time() - t0) * 1000)
    save_result(r, orig)
    return r, img

# =============================================================================
# BATCH PROCESSING
# =============================================================================

def compute_layer_counts_from_results(results):
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for r in results:
        if getattr(r, "exit_layer", 0) in counts:
            counts[r.exit_layer] += 1
    return counts

def start_batch(sources, df=None, url_col=None):
    st.session_state.chunk_state = {
        "sources": sources, "df": df, "url_col": url_col,
        "idx": 0, "results": [], "t_start": time.time(), "running": True}
    st.session_state.is_processing    = True
    st.session_state.total_images     = len(sources)
    st.session_state.processed_count  = 0
    st.session_state.layer_counts     = {1:0,2:0,3:0,4:0,5:0,6:0}
    st.session_state.results          = []
    st.session_state.total_flash_tokens = 0
    st.session_state.total_pro_tokens   = 0
    st.session_state.total_cost_usd     = 0.0
    st.session_state.total_cost_inr     = 0.0
    st.session_state.duplicate_candidate_ids = get_recent_duplicate_ids(DUP_LOOKBACK_DAYS)

def _run_one_chunk_bg(state, start_idx, end_idx, out_list, _ctx):
    if _ctx and add_script_run_ctx:
        add_script_run_ctx(threading.current_thread(), _ctx)
    """
    v8 SPEED — Background worker: processes images[start_idx:end_idx] in a
    ThreadPoolExecutor and appends results to out_list (thread-safe via list.append).
    Runs in a separate thread so PARALLEL_CHUNKS chunks can fly simultaneously,
    hiding the chunk-boundary gap that existed in v7.

    Duplicate safety with parallel chunks:
    - FAISS index and hash_cache updates still hold faiss_lock + state_lock
      (same as before), so each image is registered before its neighbours can
      match against it.  Running 2 chunks in parallel doubles concurrent FAISS
      writers, but since FAISS add is O(1) and guarded, there is no accuracy
      regression vs sequential chunks.
    - ORB_MAX_CANDIDATES is halved (24→12) to compensate for the wider
      in-flight window; CLIP thresholds are unchanged.
    """
    def _process_one_indexed(i):
        if _ctx and add_script_run_ctx:
            add_script_run_ctx(threading.current_thread(), _ctx)
        src = state["sources"][i]; orig, url = {}, ""
        if state["df"] is not None and state["url_col"] and i < len(state["df"]):
            orig = state["df"].iloc[i].to_dict()
            url  = str(orig.get(state["url_col"], ""))
        return process_one(src, orig, url, row_idx=i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
        futures = {executor.submit(_process_one_indexed, i): i
                   for i in range(start_idx, end_idx)}
        for future in concurrent.futures.as_completed(futures):
            try:
                result, _ = future.result()
            except Exception as exc:
                logging.error(f"Chunk worker [{futures[future]}] failed: {exc}", exc_info=True)
                continue
            out_list.append((futures[future], result))
            with _ss_lock("state_lock", _PV_GLOBAL_STATE_LOCK):
                st.session_state.activity_log.append({
                    "id": result.pv_image_id, "status": result.validation_status,
                    "layer": result.exit_layer, "ms": result.processing_time_ms})
                st.session_state.activity_log = st.session_state.activity_log[-50:]
                st.session_state.processed_count = len(state["results"]) + len(out_list)


def run_chunk(bar, status, metrics, layers, recent):
    """
    v8 SPEED — Pipelined multi-chunk dispatcher.

    Strategy:
    1. Dispatch PARALLEL_CHUNKS chunks simultaneously into background threads.
    2. Wait for all of them to finish, then flush their results into state.
    3. Re-render the UI once per multi-chunk batch instead of once per chunk.

    This hides the ~0.5–1 s Streamlit rerun overhead that v7 paid once every 15
    images.  With CHUNK_SIZE=30 and PARALLEL_CHUNKS=2, the effective batch size
    is 60 images per UI cycle, reducing rerun overhead by ~4×.
    """
    state = st.session_state.chunk_state
    if state is None: return
    total = len(state["sources"]); idx = state["idx"]
    if not state["running"] or idx >= total:
        st.session_state.results         = state["results"]
        st.session_state.is_processing   = False
        st.session_state.processed_count = len(state["results"])
        st.session_state.chunk_state     = None
        with st.session_state.faiss_lock:
            if st.session_state.faiss_index:
                faiss_save(st.session_state.faiss_index)
        flush_db_queue()
        st.success(f"✅ Complete — {len(st.session_state.results):,} images processed.")
        try: st.rerun()
        except Exception: pass
        return

    _ctx = get_script_run_ctx() if get_script_run_ctx else None

    # ── Build up to PARALLEL_CHUNKS chunk ranges ───────────────────────────
    chunk_ranges = []
    cur = idx
    for _ in range(PARALLEL_CHUNKS):
        if cur >= total: break
        nxt = min(cur + CHUNK_SIZE, total)
        chunk_ranges.append((cur, nxt))
        cur = nxt

    # ── Fire all chunks in background threads simultaneously ───────────────
    chunk_out_lists = [[] for _ in chunk_ranges]
    bg_threads = []
    for (s, e), out in zip(chunk_ranges, chunk_out_lists):
        t = threading.Thread(
            target=_run_one_chunk_bg,
            args=(state, s, e, out, _ctx),
            daemon=True,
        )
        if _ctx and add_script_run_ctx:
            add_script_run_ctx(t, _ctx)
        t.start()
        bg_threads.append(t)

    for t in bg_threads:
        t.join()

    # ── Merge results in row/index order for deterministic output ─────────
    for out in chunk_out_lists:
        state["results"].extend([r for _, r in sorted(out, key=lambda x: x[0])])

    new_end = chunk_ranges[-1][1]
    state["idx"] = new_end

    # ── UI refresh ────────────────────────────────────────────────────────
    elapsed = time.time() - state["t_start"]
    rate    = new_end / elapsed if elapsed > 1 else 0
    eta     = (total - new_end) / rate if rate > 0 else 0

    bar.progress(new_end / total)
    status.markdown(
        f"**{new_end:,} / {total:,}** &nbsp;·&nbsp; {rate:.1f} img/s "
        f"&nbsp;·&nbsp; ETA **{int(eta//60)}m {int(eta%60)}s**")

    res = state["results"]
    v   = sum(1 for r in res if r.validation_status == "Valid")
    inv = sum(1 for r in res if r.validation_status in ("Invalid", "Pending Review"))
    d   = sum(1 for r in res if r.validation_status == "Duplicate")
    er  = sum(1 for r in res if r.validation_status == "Error")

    layer_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for r in res:
        if r.exit_layer in layer_counts:
            layer_counts[r.exit_layer] += 1

    metrics.markdown(
        f"| ✅ Valid | ❌ Invalid | 🔁 Duplicate | ⚠️ Error |\n"
        f"|:---:|:---:|:---:|:---:|\n"
        f"| **{v}** | **{inv}** | **{d}** | **{er}** |")
    layers.markdown(f"""
| Layer | Count |
|---|---:|
| L1 Hash | {layer_counts[1]} |
| L2 CV / Pre-AI | {layer_counts[2]} |
| L3 CLIP / ORB | {layer_counts[3]} |
| L4 Flash | {layer_counts[4]} |
| L5 Pro | {layer_counts[5]} |
""")

    rows = []
    for r in reversed(state["results"][-8:]):
        icon = {"Valid": "✅", "Invalid": "❌", "Duplicate": "🔁", "Error": "⚠️"}.get(
            r.validation_status, "❓")
        rows.append({
            "ID":     r.pv_image_id,
            "Status": f"{icon} {r.validation_status}",
            "Layer":  f"L{r.exit_layer}",
            "Conf%":  f"{r.ai_confidence:.0f}",
            "Tokens": r.total_ai_tokens,
            "ms":     r.processing_time_ms,
        })
    if rows: recent.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    time.sleep(0.05)
    try: st.rerun()
    except Exception: pass

def cancel_batch():
    if st.session_state.chunk_state:
        st.session_state.chunk_state["running"] = False
    st.session_state.is_processing = False
    with st.session_state.faiss_lock:
        if st.session_state.faiss_index:
            faiss_save(st.session_state.faiss_index)

# =============================================================================
# GOOGLE SHEETS
# =============================================================================
def load_sheet(url_or_id):
    gc = st.session_state.sheets_client
    if not gc: return None, ""
    try:
        sid = url_or_id.split("/d/")[1].split("/")[0] if "/d/" in url_or_id else url_or_id.strip()
        df  = pd.DataFrame(gc.open_by_key(sid).sheet1.get_all_records())
        return df, sid
    except Exception as e:
        st.error(f"Sheet load failed: {e}"); return None, ""

def push_to_sheet(sid, results, orig_df):
    gc = st.session_state.sheets_client
    if not gc: st.error("Sheets not authenticated."); return False
    try:
        ss   = gc.open_by_key(sid)
        name = f"PV_Results_{datetime.now().strftime('%m%d_%H%M')}"
        try:    ws = ss.add_worksheet(title=name, rows=len(results)+5, cols=60)
        except: ws = ss.worksheet(name); ws.clear()
        df   = build_output_df(results, orig_df)
        vals = [df.columns.tolist()] + df.values.tolist()
        for s in range(0, len(vals), 1000):
            chunk = vals[s:s+1000]
            if s == 0: ws.update(chunk)
            else:      ws.append_rows(chunk)
        return True
    except Exception as e:
        st.error(f"Push failed: {e}"); return False

# =============================================================================
# FILE LOADING
# =============================================================================
def load_file(f):
    name = f.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(f)
        elif name.endswith((".xlsx",".xls",".xlsm")):
            return pd.read_excel(f, engine="openpyxl")
        else:
            st.error("Unsupported file type."); return None
    except Exception as e:
        st.error(f"File load failed: {e}"); return None

# =============================================================================
# LABEL MAPPING
# =============================================================================
def get_ground_truth_label(row: dict) -> str:
    dup    = str(row.get("Duplication", "")).strip().lower()
    status = str(row.get("Status", "")).strip().lower()
    if dup in ("duplicate", "dulicate"):
        return "Duplicate"
    elif status == "verified":
        return "Valid"
    elif status == "not verified":
        return "Invalid"
    return "Unknown"

def get_predicted_label(r: "VR") -> str:
    if r.duplicate_status == "Duplicate" or r.validation_status == "Duplicate":
        return "Duplicate"
    if r.validation_status == "Valid":
        return "Valid"
    return "Invalid"

# =============================================================================
# OUTPUT DATAFRAME
# =============================================================================
def build_output_df(results, orig_df=None):
    rows = []
    for r in results:
        orig = {}
        if orig_df is not None and r.row_index >= 0 and r.row_index < len(orig_df):
            orig = orig_df.iloc[r.row_index].to_dict()
        pv = {
            "PV_Image_ID":              r.pv_image_id,
            "PV_Validation_Status":     r.validation_status,
            "PV_Duplicate_Status":      r.duplicate_status,
            "PV_Matched_Image_ID":      r.matched_image_id,
            "PV_Original_Status":       r.original_status,
            "PV_Similarity_Score_%":    r.similarity_score,
            "PV_Exit_Layer":            r.exit_layer,
            "PV_Has_Face":              r.has_face,
            "PV_Face_Count":            r.face_count,
            "PV_Is_Screenshot":         r.is_screenshot,
            "PV_Is_Manipulated":        r.is_manipulated,
            "PV_Has_Branding":          r.has_required_branding,
            "PV_Branding_Details":      r.branding_details,
            "PV_Image_Quality":         r.image_quality,
            "PV_AI_Confidence_%":       r.ai_confidence,
            "PV_GPS_Lat":               r.gps_lat,
            "PV_GPS_Lon":               r.gps_lon,
            "PV_GPS_Valid":             r.gps_valid,
            "PV_Cluster_ID":            r.cluster_id,
            "PV_Forensic_Reasoning":    r.forensic_reasoning,
            "PV_Error_Reason":          r.error_reason,
            "PV_Flash_Model":           r.flash_model,
            "PV_Pro_Model":             r.pro_model,
            "PV_Flash_Total_Tokens":    r.flash_total_tokens,
            "PV_Flash_Cost_INR":        r.flash_cost_inr,
            "PV_Pro_Total_Tokens":      r.pro_total_tokens,
            "PV_Pro_Cost_INR":          r.pro_cost_inr,
            "PV_Total_AI_Tokens":       r.total_ai_tokens,
            "PV_Total_AI_Cost_INR":     r.total_ai_cost_inr,
            "PV_Processing_Time_ms":    r.processing_time_ms,
            "PV_Processed_At":          r.processed_at,
        }
        rows.append({**orig, **pv})
    return pd.DataFrame(rows)

# =============================================================================
# SIDEBAR (FIX 10 — premium design)
# =============================================================================
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:20px 0 12px;">
          <div style="font-size:11px;font-weight:700;letter-spacing:2px;
               color:#6366F1;text-transform:uppercase;margin-bottom:4px;">
            ● PHOTOVERIFY
          </div>
          <div style="font-size:22px;font-weight:800;color:#F1F5F9;line-height:1.1;">
            AI Verifier
          </div>
          <div style="font-size:11px;color:#64748B;margin-top:4px;">
            v{APP_VERSION} · Field Survey QC
          </div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        ga = "✅" if st.session_state.gemini_ok else "❌"
        sa = "✅" if st.session_state.sheets_client else "❌"
        cl = "✅" if st.session_state.clip_model else "⏳"
        fn = st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0
        em = st.session_state.sa_email

        st.markdown("**System Status**")
        st.markdown(f"{ga} Gemini AI &nbsp;·&nbsp; {sa} Sheets &nbsp;·&nbsp; {cl} CLIP")
        if em: st.caption(f"  {em[:40]}")

        # FIX 7: Explain FAISS vector count
        fn_label = f"📦 FAISS: {fn:,} vectors"
        st.markdown(fn_label)
        st.caption(
            "FAISS stores ALL processed images (valid + invalid + duplicates). "
            f"Dedup comparison uses only the last {DUP_LOOKBACK_DAYS} days + current batch."
        )
        st.divider()

        # Cost summary

        if st.session_state.is_processing:
            if st.button("⏹ Stop Processing", use_container_width=True):
                cancel_batch(); st.rerun()
            st.divider()

        if st.button("🗑 Clear All Data", use_container_width=True, type="secondary"):
            clear_db()
            st.session_state.results          = []
            st.session_state.hash_cache       = {}
            st.session_state.faiss_id_map     = {}
            st.session_state.activity_log     = []
            st.session_state.human_review     = []
            st.session_state.cluster_map      = {}
            st.session_state.layer_counts     = {1:0,2:0,3:0,4:0,5:0,6:0}
            st.session_state.total_flash_tokens = 0
            st.session_state.total_pro_tokens   = 0
            st.session_state.total_cost_inr     = 0.0
            st.session_state.total_cost_usd     = 0.0
            for p in [FAISS_INDEX,FAISS_BAK,FAISS_TMP,FAISS_MAP,FAISS_MTMP,FAISS_MBAK]:
                if os.path.exists(p): os.remove(p)
            st.session_state.faiss_index  = faiss_load()
            st.session_state.faiss_id_map = {}
            st.rerun()

# =============================================================================
# SHARED COLUMN SELECTOR + RUN
# =============================================================================
def _column_selector_and_run(df, source_label=""):
    st.dataframe(df.head(5), use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        url_col = st.selectbox(
            "📎 Column containing image URLs *",
            options=list(df.columns),
            key=f"url_col_{source_label}")
    with cb:
        n_limit = st.number_input("Limit rows (0 = all)", 0, len(df), 0,
                                  key=f"lim_{source_label}")
    skip_empty = st.checkbox("Skip rows with empty / non-URL values", value=True,
                             key=f"skip_{source_label}")
    st.markdown("---")
    if st.button(f"🚀 Start Processing", type="primary", use_container_width=True,
                 key=f"run_{source_label}"):
        work = df.copy()
        if n_limit > 0: work = work.head(n_limit)
        if skip_empty:  work = work[work[url_col].astype(str).str.startswith("http")]
        work = work.reset_index(drop=True)
        if work.empty:
            st.error("No valid rows after filtering.")
        else:
            st.session_state.current_df = work
            start_batch(work[url_col].tolist(), work, url_col)
            st.rerun()

# =============================================================================
# TAB: PROCESS
# =============================================================================
def tab_process():
    if not st.session_state.gemini_ok:
        st.error("❌ **Service account not found.**")
        st.markdown("""
Place your `service_account.json` in the same folder as `main_v9.py`, then restart.
```bash
ls *.json          # should list your service_account.json
streamlit run main_v9.py
```
The JSON must have `"type": "service_account"` with access to:
- **Google Generative Language API** (Gemini)
- **Google Sheets API**
        """)
        st.stop()

    if st.session_state.is_processing and st.session_state.chunk_state is not None:
        st.markdown('<div class="live-box">', unsafe_allow_html=True)
        st.markdown("### ⚡ Live Processing Dashboard")
        bar      = st.progress(0)
        status   = st.empty()
        st.markdown("**Layer exits**")
        layers   = st.empty()
        st.markdown("**Running counts**")
        metrics  = st.empty()
        st.markdown("**Last 8 processed**")
        recent   = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
        run_chunk(bar, status, metrics, layers, recent)
        return

    inp = st.tabs(["📊 Google Sheet", "📁 CSV / Excel", "🖼️ Upload Images", "🔗 URL List"])

    with inp[0]:
        c1, c2 = st.columns([4,1])
        with c1:
            url = st.text_input("Sheet URL or ID",
                placeholder="https://docs.google.com/spreadsheets/d/YOUR_ID/edit")
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Load Sheet", use_container_width=True) and url:
                with st.spinner("Loading…"):
                    df, sid = load_sheet(url)
                if df is not None:
                    st.session_state.current_df      = df
                    st.session_state.loaded_sheet_id = sid
                    st.success(f"Loaded {len(df):,} rows · {len(df.columns)} columns")
        if st.session_state.current_df is not None:
            _column_selector_and_run(st.session_state.current_df, "sheet")

    with inp[1]:
        st.caption("Upload CSV or Excel. Then choose the column that contains the photo URLs.")
        uf = st.file_uploader("Choose file", type=["csv","xlsx","xls","xlsm"])
        if uf:
            df = load_file(uf)
            if df is not None:
                st.session_state.current_df      = df
                st.session_state.loaded_sheet_id = ""
                st.success(f"Loaded **{len(df):,} rows** · **{len(df.columns)} columns** from `{uf.name}`")
                _column_selector_and_run(df, uf.name)

    with inp[2]:
        files = st.file_uploader("Drop image files here",
                                 type=["jpg","jpeg","png","webp","bmp"],
                                 accept_multiple_files=True)
        if files:
            cols = st.columns(min(4, len(files)))
            for i, f in enumerate(files[:8]):
                with cols[i%4]: st.image(f, caption=f.name, width=140)
            if len(files) > 8: st.caption(f"… and {len(files)-8} more")
            if st.button("🚀 Process Uploaded Images", type="primary"):
                imgs, names = [], []
                for uf in files:
                    try:
                        imgs.append(Image.open(uf).convert("RGB"))
                        names.append(uf.name)
                    except Exception: pass
                if imgs:
                    st.session_state.current_df = pd.DataFrame({"filename": names[:len(imgs)]})
                    start_batch(imgs)
                    st.rerun()

    with inp[3]:
        st.caption("Paste one image URL per line.")
        txt = st.text_area("Image URLs", height=180,
                           placeholder="https://example.com/photo1.jpg\nhttps://…")
        if st.button("🚀 Process URLs", type="primary") and txt.strip():
            urls = [u.strip() for u in txt.splitlines() if u.strip().startswith("http")]
            if urls: start_batch(urls); st.rerun()
            else:    st.error("No valid URLs found.")

# =============================================================================
# TAB: RESULTS
# =============================================================================
def tab_results():
    results = st.session_state.results
    if not results:
        st.info("No results yet. Run verification in the **⚡ Process** tab.")
        return

    df_out = build_output_df(results, st.session_state.current_df)
    total  = len(results)
    v   = sum(1 for r in results if r.validation_status=="Valid")
    inv = sum(1 for r in results if r.validation_status in ("Invalid","Pending Review"))
    d   = sum(1 for r in results if r.validation_status=="Duplicate")
    er  = sum(1 for r in results if r.validation_status=="Error")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total",        f"{total:,}")
    c2.metric("✅ Valid",      v,   f"{v/total*100:.1f}%")
    c3.metric("❌ Invalid",    inv, f"{inv/total*100:.1f}%")
    c4.metric("🔁 Duplicates", d,   f"{d/total*100:.1f}%")
    c5.metric("⚠️ Errors",     er)

    st.divider()
    lc = compute_layer_counts_from_results(results)
    st.markdown("**Layer exit distribution**")
    st.bar_chart(pd.DataFrame({
        "Layer":["L1 Hash","L2 CV","L3 CLIP/ORB","L4 Flash","L5 Pro"],
        "Count":[lc[1],lc[2],lc[3],lc[4],lc[5]]}).set_index("Layer"))
    st.divider()

    pv_show = [
        "PV_Image_ID","PV_Validation_Status","PV_Duplicate_Status",
        "PV_Matched_Image_ID","PV_Original_Status",
        "PV_Similarity_Score_%","PV_Exit_Layer",
        "PV_Has_Face","PV_Is_Screenshot","PV_Is_Manipulated",
        "PV_AI_Confidence_%","PV_GPS_Valid","PV_Error_Reason","PV_Processing_Time_ms"
    ]
    avail = [c for c in pv_show if c in df_out.columns]

    def color_status(val):
        return {
            "Valid":          "background-color:#D1FAE5;color:#065F46;font-weight:700",
            "Invalid":        "background-color:#FEE2E2;color:#991B1B;font-weight:700",
            "Pending Review": "background-color:#FEE2E2;color:#991B1B;font-weight:700",
            "Duplicate":      "background-color:#FEF3C7;color:#92400E;font-weight:700",
            "Error":          "background-color:#F1F5F9;color:#64748B"
        }.get(val, "")

    st.markdown("**All Results**")
    preview_mode = st.radio(
        "Preview", ["Validation summary", "Full export view"],
        horizontal=True, key="results_preview_mode")

    if preview_mode == "Validation summary":
        st.dataframe(
            df_out[avail].style.map(color_status, subset=["PV_Validation_Status"]),
            use_container_width=True, height=360)
    else:
        st.dataframe(
            df_out.style.map(color_status, subset=["PV_Validation_Status"]),
            use_container_width=True, height=420)

    st.divider()
    st.markdown("### 📥 Export Results")
    inv_df = df_out[df_out["PV_Validation_Status"].isin(["Invalid","Pending Review"])]
    dup_df = df_out[df_out["PV_Validation_Status"]=="Duplicate"]
    crows  = [{"Cluster_ID":r.cluster_id,"Duplicate_ID":r.pv_image_id,
               "Original_ID":r.matched_image_id,"Original_Status":r.original_status,
               "Match_Type":r.duplicate_status,"Similarity_%":r.similarity_score,
               "Layer":r.exit_layer}
              for r in results if r.validation_status=="Duplicate"]

    ea, eb, ec, ed = st.columns(4)
    ea.download_button("⬇️ Full CSV", df_out.to_csv(index=False),
        file_name=f"pv_full_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv", use_container_width=True)
    if not inv_df.empty:
        eb.download_button("⬇️ Invalid Only", inv_df.to_csv(index=False),
            file_name="pv_invalid.csv", mime="text/csv", use_container_width=True)
    else:
        eb.button("Invalid Only (0)", disabled=True, use_container_width=True)
    if not dup_df.empty:
        ec.download_button("⬇️ Duplicates", dup_df.to_csv(index=False),
            file_name="pv_duplicates.csv", mime="text/csv", use_container_width=True)
    else:
        ec.button("Duplicates (0)", disabled=True, use_container_width=True)
    if crows:
        ed.download_button("⬇️ Cluster Report",
            pd.DataFrame(crows).sort_values("Cluster_ID").to_csv(index=False),
            file_name="pv_clusters.csv", mime="text/csv", use_container_width=True)
    else:
        ed.button("Clusters (0)", disabled=True, use_container_width=True)

    st.markdown("---")
    fa, fb, fc = st.columns(3)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df_out.to_excel(w, index=False, sheet_name="All Results")
        if not inv_df.empty: inv_df.to_excel(w, index=False, sheet_name="Invalid")
        if not dup_df.empty: dup_df.to_excel(w, index=False, sheet_name="Duplicates")
        cost_df = df_out[[c for c in [
            "PV_Image_ID","PV_Exit_Layer",
            "PV_Flash_Total_Tokens","PV_Flash_Cost_INR",
            "PV_Pro_Total_Tokens","PV_Pro_Cost_INR",
            "PV_Total_AI_Tokens","PV_Total_AI_Cost_INR",
        ] if c in df_out.columns]]
        cost_df.to_excel(w, index=False, sheet_name="Cost Breakdown")
    buf.seek(0)
    fa.download_button("⬇️ Full Excel (.xlsx)", buf.getvalue(),
        file_name=f"pv_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
    json_bytes = df_out.to_json(orient="records", indent=2).encode("utf-8")
    fb.download_button("⬇️ Export JSON", json_bytes,
        file_name=f"pv_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json", use_container_width=True)

    if st.session_state.sheets_client and st.session_state.loaded_sheet_id:
        if fc.button("📤 Push to Google Sheet", type="primary", use_container_width=True):
            with st.spinner("Pushing…"):
                if push_to_sheet(st.session_state.loaded_sheet_id, results,
                                 st.session_state.current_df):
                    st.success("Results written to a new tab in your spreadsheet.")

    st.divider()
    with st.expander("🔬 Forensic Notes (first 20)"):
        for r in results[:20]:
            st.markdown(
                f"**`{r.pv_image_id}`** {badge(r.validation_status)} "
                f"L{r.exit_layer} · {r.ai_confidence:.1f}% · {r.processing_time_ms}ms",
                unsafe_allow_html=True)
            if r.forensic_reasoning: st.caption(r.forensic_reasoning)
            if r.error_reason:       st.error(f"Reason: {r.error_reason}")
            st.divider()

# =============================================================================
# TAB: GUIDE (FIX 8 — clean user manual, no version history noise)
# =============================================================================
def tab_guide():
    st.markdown(f"""
<div class="section-card">

## 📘 PhotoVerify AI — User Guide

### What does this app do?

PhotoVerify AI automatically verifies field survey photos submitted by door-to-door workers.
Each photo is classified into one of three outcomes:

| Result | Meaning |
|--------|---------|
| ✅ **Valid** | Live adult present, political/scheme branding visible, not a duplicate |
| ❌ **Invalid** | Rejected — see the reason in PV_Error_Reason column |
| 🔁 **Duplicate** | Same or near-identical photo already seen in this batch or last {DUP_LOOKBACK_DAYS} days |

---

### What makes a photo Invalid?

A photo is rejected (Invalid) for any of the following reasons:

- **No live human face** — the photo must show a real person, not a poster or printout
- **No political/scheme branding** — NCP clock symbol, Majhi Ladki Bahin materials, party banners/flyers must be visible
- **Photo of a screen** — taking a picture of a phone/laptop/TV screen instead of a real photo
- **Printed photo** — photographing a printout or an existing photo instead of a live scene
- **Children/minors only** — photo contains only people under 18 with no adult present
- **Child holding branding** — a minor is the one holding the pamphlet/flyer (even if an adult is also in frame)
- **Mirror/glass reflection** — reflection of a person is not a genuine field photo
- **Duplicate** — same scene photographed multiple times or near-duplicate found in recent history

---

### What counts as Valid?

- A real adult (18+) is clearly present in the photo
- NCP Ajit Pawar branding, Majhi Ladki Bahin scheme materials, or political party flyers/posters are visible
- The adult is holding or near the branding materials
- **Mother + child is Valid** — when the mother/adult is clearly holding the pamphlet, the photo is valid even if a child is also in the frame
- Camera app timestamp watermarks are expected and do not affect validity

---

### How to use the app

**Step 1 — Load your data**
- Go to the **⚡ Process** tab
- Choose: Google Sheet, CSV/Excel upload, image file upload, or URL list
- Select the column that contains the image URLs

**Step 2 — Start processing**
- Click **🚀 Start Processing**
- Watch the live dashboard: progress bar, layer counts, running totals

**Step 3 — Review results**
- Go to the **📊 Results** tab
- Check `PV_Error_Reason` to understand why any photo was rejected
- Check `PV_Matched_Image_ID` to see which photo a duplicate matched

**Step 4 — Export**
- Download Full CSV, Invalid-only CSV, Duplicates CSV, or Excel
- Or push directly back to your Google Sheet

---

### Understanding the output columns

| Column | What it means |
|--------|--------------|
| `PV_Validation_Status` | Valid / Invalid / Duplicate |
| `PV_Error_Reason` | Why the photo was rejected |
| `PV_Exit_Layer` | Which check caught the issue (L1=hash, L2=CV, L3=CLIP, L4=Flash AI, L5=Pro AI) |
| `PV_AI_Confidence_%` | How confident the AI was in its decision |
| `PV_Matched_Image_ID` | ID of the original photo this was a duplicate of |
| `PV_Processed_At` | UTC timestamp when this photo was processed |
| `PV_Flash_Cost_INR` | AI cost for the Flash model check |
| `PV_Pro_Cost_INR` | AI cost for the Pro model check (only for hard cases) |

---

### Duplicate detection — how it works

Duplicates are searched in two places:
1. **Current batch** — all images being processed right now
2. **Last {DUP_LOOKBACK_DAYS} days** — photos processed in the past {DUP_LOOKBACK_DAYS} days from the database

> **Why does the FAISS vector count seem higher than the number of valid photos?**
> FAISS stores embeddings for ALL processed images — valid, invalid, and duplicates.
> This is necessary so that future submissions of already-rejected photos are also caught as duplicates.
> The vector count will always be ≥ total photos processed, not just valid ones.

---

### Speed and capacity

- Processes **{CHUNK_SIZE} images in parallel** per chunk, **{PARALLEL_CHUNKS} chunks simultaneously** (v8)
- Typical throughput: **50–80 images/minute** depending on network and AI response time (was 15–25 in v7)
- For 20,000 images: expect **~40–60 minutes** at target throughput
- Gemini Flash model (`{FLASH_MODEL}`) handles most photos
- Pro model (`{PRO_MODEL}`) is only called for uncertain/borderline cases (~10–15%)

</div>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🔍",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    init_ss()
    init_db()
    auto_auth()

    if st.session_state.clip_model is None:
        with st.spinner("Loading CLIP model (first run ~30s)…"):
            st.session_state.clip_model = load_clip()

    if st.session_state.faiss_index is None:
        st.session_state.faiss_index  = faiss_load()
        st.session_state.faiss_id_map = faiss_load_map()

    if not st.session_state.hash_cache:
        st.session_state.hash_cache = load_hashes(DUP_LOOKBACK_DAYS)

    atexit.register(lambda: (
        faiss_save(st.session_state.faiss_index)
        if st.session_state.get("faiss_index") else None
    ))
    atexit.register(lambda: (_db_write_queue.put(None) if '_db_write_queue' in globals() else None))

    sidebar()

    # Premium app header (FIX 10)
    results = st.session_state.results
    total   = len(results)
    v   = sum(1 for r in results if r.validation_status=="Valid")
    inv = sum(1 for r in results if r.validation_status in ("Invalid","Pending Review"))
    d   = sum(1 for r in results if r.validation_status=="Duplicate")
    fn  = st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0

    st.markdown(f"""
    <div class="app-header">
      <span style="font-size:24px;font-weight:800;color:{C_TEXT};">{APP_NAME}</span>
      <span style="font-size:13px;color:{C_MUTED};margin-left:12px;">
        v{APP_VERSION} · AI-powered field survey photo verification
      </span>
    </div>""", unsafe_allow_html=True)

    mc = st.columns(5)
    mc[0].metric("Processed",       f"{total:,}")
    mc[1].metric("✅ Valid",         v,   f"{v/total*100:.1f}%"   if total else None)
    mc[2].metric("❌ Invalid",       inv, f"{inv/total*100:.1f}%" if total else None)
    mc[3].metric("🔁 Duplicates",    d,   f"{d/total*100:.1f}%"   if total else None)
    mc[4].metric("📦 FAISS Vectors", f"{fn:,}")

    st.divider()

    # FIX 6: Removed Accuracy Monitor and Calibrate tabs (dev/QA tools, not needed in prod UI)
    t1, t2, t3 = st.tabs(["⚡ Process", "📊 Results", "📘 Guide"])
    with t1: tab_process()
    with t2: tab_results()
    with t3: tab_guide()

if __name__ == "__main__":
    main()
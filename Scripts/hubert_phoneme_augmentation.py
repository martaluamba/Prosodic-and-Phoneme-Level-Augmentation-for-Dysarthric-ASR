# -*- coding: utf-8 -*-
"""
HuBERT Fine-tuning on UASpeech with Phoneme-Level Data Augmentation
=====================================================================
Colab-ready — paste each CELL block into a separate Colab cell.

Pipeline:
  transcript → IPA (espeak-ng) → phoneme augmentation
  (substitution / deletion / prolongation) → TTS synthesis
  → augmented waveform → HuBERT CTC fine-tuning

Authors: Group 4 (adapted for phoneme-level augmentation)
"""

# =====================================================================
# CELL 1 — System & Python installs
# Run this first, then restart runtime if prompted.
# =====================================================================
"""
!apt-get install -y espeak-ng libespeak-ng-dev
!pip install torch torchaudio transformers librosa jiwer kagglehub
!pip install phonemizer
!pip install TTS
"""

# =====================================================================
# CELL 2 — Download UASpeech dataset from Kaggle
# =====================================================================
"""
import kagglehub
from pathlib import Path
import os

path = kagglehub.dataset_download("aryashah2k/noise-reduced-uaspeech-dysarthria-dataset")
print("Dataset root:", path)

# Inspect folder layout
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 3:
        for f in files[:5]:
            print(f'{indent}  {f}')
"""

# =====================================================================
# CELL 3 — All imports
# =====================================================================

import gc
import os
import re
import csv
import random
import tempfile
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader
from transformers import (
    HubertForCTC,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup,
)
from jiwer import wer, cer

# Phonemizer
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

# Coqui TTS
from TTS.api import TTS as CoquiTTS

print("All imports successful.")

# =====================================================================
# CELL 4 — Global config
# =====================================================================

# ── Paths ─────────────────────────────────────────────────────────────
# Replace with your actual kagglehub path after Cell 2
# path = kagglehub.dataset_download("aryashah2k/noise-reduced-uaspeech-dysarthria-dataset")
DATASET_ROOT = Path(path)
OUTPUT_DIR   = "./checkpoints_phoneme"

# ── Training hyperparameters ──────────────────────────────────────────
POLICY           = "phoneme"   # none | prosodic | phoneme
EPOCHS           = 3
BATCH_SIZE       = 1           # Keep at 1 for hubert-large; use GRAD_ACCUM for effective batch
GRAD_ACCUM_STEPS = 4           # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
LR               = 1e-4
VAL_SPLIT        = 0.15
NUM_WORKERS      = 0           # 0 is safest in Colab (avoids multiprocess OOM)
SEED             = 42
TARGET_SR        = 16_000      # HuBERT expects 16 kHz

# ── Model ─────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/hubert-large-ls960-ft"
# If you hit OOM even after the memory fixes, switch to:
# MODEL_NAME = "facebook/hubert-base-ls960-ft"

# ── Phoneme augmentation probabilities ───────────────────────────────
SUB_PROB = 0.15   # Probability of substituting each eligible phoneme
DEL_PROB = 0.10   # Probability of deleting each consonant
PRO_PROB = 0.10   # Probability of prolonging each vowel/sonorant

# =====================================================================
# CELL 5 — UASpeech word-ID vocabulary
#
# The Kaggle dataset has NO transcript files.
# Labels are encoded in the filename itself:
#   <speaker>_<block>_<word_id>_<mic>.wav
#   e.g.  CF02_B1_UW1_M7.wav  →  word_id = "UW1" → "academic"
# =====================================================================

UW = {
    "UW1":  "academic",   "UW2":  "accuse",     "UW3":  "achieve",
    "UW4":  "acoustics",  "UW5":  "across",     "UW6":  "action",
    "UW7":  "adjust",     "UW8":  "adopt",      "UW9":  "algebra",
    "UW10": "allege",     "UW11": "allies",     "UW12": "alphabet",
    "UW13": "amber",      "UW14": "amend",      "UW15": "ammonia",
    "UW16": "ancient",    "UW17": "annex",      "UW18": "anoint",
    "UW19": "antenna",    "UW20": "antique",    "UW21": "apex",
    "UW22": "aqua",       "UW23": "arctic",     "UW24": "ardent",
    "UW25": "arsenal",    "UW26": "ascend",     "UW27": "asphalt",
    "UW28": "assign",     "UW29": "astute",     "UW30": "atone",
    "UW31": "attic",      "UW32": "audit",      "UW33": "augment",
    "UW34": "austere",    "UW35": "autopsy",    "UW36": "avant",
    "UW37": "avert",      "UW38": "avid",       "UW39": "avocado",
    "UW40": "axiom",      "UW41": "azure",      "UW42": "ballot",
    "UW43": "banquet",    "UW44": "baroque",    "UW45": "beckon",
    "UW46": "bequeath",   "UW47": "bizarre",    "UW48": "blizzard",
    "UW49": "bogus",      "UW50": "bonanza",    "UW51": "bravado",
    "UW52": "brevity",    "UW53": "brim",       "UW54": "brochure",
    "UW55": "brooch",     "UW56": "buffet",     "UW57": "bulwark",
    "UW58": "bungle",     "UW59": "bureau",     "UW60": "burnish",
    "UW61": "cadence",    "UW62": "cajole",     "UW63": "calyx",
    "UW64": "camber",     "UW65": "canopy",     "UW66": "canvas",
    "UW67": "capsule",    "UW68": "carjack",    "UW69": "carnage",
    "UW70": "casserole",  "UW71": "catalyst",   "UW72": "caustic",
    "UW73": "cavalry",    "UW74": "cellar",     "UW75": "census",
    "UW76": "ceramic",    "UW77": "chasm",      "UW78": "chronic",
    "UW79": "cipher",     "UW80": "citrus",     "UW81": "clamor",
    "UW82": "clarity",    "UW83": "clemency",   "UW84": "cliche",
    "UW85": "cobalt",     "UW86": "coerce",     "UW87": "cognac",
    "UW88": "collage",    "UW89": "colossal",   "UW90": "comrade",
    "UW91": "conduit",    "UW92": "conifer",    "UW93": "consul",
    "UW94": "contour",    "UW95": "convex",     "UW96": "convoy",
    "UW97": "copious",    "UW98": "corona",     "UW99": "cortex",
    "UW100": "cosmos",
}

CW = {
    "CW1":  "a",         "CW2":  "able",      "CW3":  "about",
    "CW4":  "after",     "CW5":  "again",     "CW6":  "air",
    "CW7":  "all",       "CW8":  "also",      "CW9":  "an",
    "CW10": "and",       "CW11": "another",   "CW12": "any",
    "CW13": "are",       "CW14": "as",        "CW15": "ask",
    "CW16": "at",        "CW17": "away",      "CW18": "back",
    "CW19": "be",        "CW20": "because",   "CW21": "been",
    "CW22": "before",    "CW23": "big",       "CW24": "boy",
    "CW25": "but",       "CW26": "by",        "CW27": "call",
    "CW28": "came",      "CW29": "can",       "CW30": "come",
    "CW31": "could",     "CW32": "day",       "CW33": "did",
    "CW34": "do",        "CW35": "does",      "CW36": "don't",
    "CW37": "down",      "CW38": "each",      "CW39": "end",
    "CW40": "even",      "CW41": "every",     "CW42": "find",
    "CW43": "first",     "CW44": "for",       "CW45": "from",
    "CW46": "get",       "CW47": "give",      "CW48": "go",
    "CW49": "good",      "CW50": "great",     "CW51": "had",
    "CW52": "has",       "CW53": "have",      "CW54": "he",
    "CW55": "her",       "CW56": "here",      "CW57": "him",
    "CW58": "his",       "CW59": "how",       "CW60": "i",
    "CW61": "if",        "CW62": "in",        "CW63": "into",
    "CW64": "is",        "CW65": "it",        "CW66": "its",
    "CW67": "just",      "CW68": "know",      "CW69": "large",
    "CW70": "last",      "CW71": "left",      "CW72": "like",
    "CW73": "little",    "CW74": "long",      "CW75": "look",
    "CW76": "made",      "CW77": "make",      "CW78": "man",
    "CW79": "many",      "CW80": "may",       "CW81": "me",
    "CW82": "more",      "CW83": "most",      "CW84": "my",
    "CW85": "name",      "CW86": "new",       "CW87": "no",
    "CW88": "not",       "CW89": "now",       "CW90": "of",
    "CW91": "on",        "CW92": "one",       "CW93": "or",
    "CW94": "other",     "CW95": "our",       "CW96": "out",
    "CW97": "over",      "CW98": "own",       "CW99": "part",
    "CW100": "people",
}

C = {
    "C1":  "backspace",  "C2":  "cancel",     "C3":  "caps lock",
    "C4":  "close",      "C5":  "copy",       "C6":  "cut",
    "C7":  "delete",     "C8":  "end",        "C9":  "enter",
    "C10": "escape",     "C11": "home",       "C12": "insert",
    "C13": "next",       "C14": "open",       "C15": "page down",
    "C16": "page up",    "C17": "paste",      "C18": "print",
    "C19": "save",
}

D = {
    "D1": "one",   "D2": "two",   "D3": "three",
    "D4": "four",  "D5": "five",  "D6": "six",
    "D7": "seven", "D8": "eight", "D9": "nine",
}

L = {f"L{c.upper()}": c for c in "abcdefghijklmnopqrstuvwxyz"}

UASPEECH_VOCAB: Dict[str, str] = {**UW, **CW, **C, **D, **L}


def word_id_to_text(word_id: str) -> Optional[str]:
    return UASPEECH_VOCAB.get(word_id.upper())


# =====================================================================
# CELL 6 — Phoneme-level augmentation engine
# =====================================================================

# ── Dysarthric substitution table ────────────────────────────────────
# Based on Xiong et al. (2019) and TORGO/UASpeech clinical literature.
# Most common patterns:
#   - Fricative stopping: /s/→/t/, /θ/→/t/ (very common)
#   - Liquid simplification: /r/→/w/, /l/→/w/
#   - Velar fronting: /k/→/t/, /g/→/d/
#   - Affricate reduction: /tʃ/→/t/
#   - Empty string "" = deletion of that phoneme

DYSARTHRIC_SUBSTITUTIONS: Dict[str, List[str]] = {
    # Fricatives → stops (stopping) or deletion
    "s":  ["t", "θ", ""],
    "z":  ["d", "s", ""],
    "ʃ":  ["s", "tʃ", ""],
    "ʒ":  ["dʒ", "ʃ", ""],
    "f":  ["p", "v", ""],
    "v":  ["b", "f", ""],
    "θ":  ["t", "d", "f"],     # th-stopping: most documented pattern
    "ð":  ["d", "v", ""],

    # Affricates → stops
    "tʃ": ["t", "ʃ"],
    "dʒ": ["d", "ʒ"],

    # Liquids → glides or deletion
    "r":  ["w", "l", ""],
    "l":  ["w", "r", ""],

    # Velar stops → alveolar (fronting)
    "k":  ["t", "g"],
    "g":  ["d", "k"],

    # Nasals — occasionally weakened
    "ŋ":  ["n", ""],
}

# IPA consonant set for deletion targeting
IPA_CONSONANTS = set("pbtdkɡfvszʃʒθðmnŋlrwjhʔtʃdʒ")

# IPA sonorants + vowels for prolongation targeting
IPA_SONORANTS = set("mnŋlrwjaeiouæɑɒɔəɛɜɪʊʌː")


def ipa_to_list(ipa_str: str) -> List[str]:
    """
    Split an IPA string into a list of phoneme tokens.
    Handles digraphs (tʃ, dʒ) before single characters.
    Strips spaces and keeps stress markers attached to their phoneme.
    """
    # Match digraphs first, then single IPA characters with optional stress/length
    pattern = r"(tʃ|dʒ|[ˈˌ]?[a-zæɑɒɔəɛɜɪʊʌθðʃʒŋɹɾʔːɡ]+)"
    tokens = re.findall(pattern, ipa_str)
    return [t for t in tokens if t.strip()]


def substitute_phonemes(phoneme_list: List[str], prob: float = SUB_PROB) -> List[str]:
    """
    Randomly substitute phonemes following the dysarthric table.
    Empty-string substitutions act as deletions.

    Args:
        phoneme_list: Input IPA phoneme tokens.
        prob: Per-phoneme probability of substitution.

    Returns:
        Augmented phoneme list (may be shorter due to deletion substitutions).
    """
    result = []
    n_substituted = 0
    for ph in phoneme_list:
        key = ph.lstrip("ˈˌ")  # strip stress markers for table lookup
        if key in DYSARTHRIC_SUBSTITUTIONS and random.random() < prob:
            replacement = random.choice(DYSARTHRIC_SUBSTITUTIONS[key])
            if replacement:
                result.append(replacement)
                n_substituted += 1
            # else: phoneme deleted via empty-string substitution
        else:
            result.append(ph)
    return result


def delete_phonemes(phoneme_list: List[str], prob: float = DEL_PROB) -> List[str]:
    """
    Randomly delete consonants to mimic phoneme omissions in dysarthric speech.
    Vowels are never deleted (vowel omission would make speech unintelligible).

    Args:
        phoneme_list: Input IPA phoneme tokens.
        prob: Per-consonant probability of deletion.

    Returns:
        Filtered phoneme list.
    """
    result = []
    for ph in phoneme_list:
        key = ph.lstrip("ˈˌ")
        # Only delete consonants, never vowels
        if key in IPA_CONSONANTS and random.random() < prob:
            pass  # deleted
        else:
            result.append(ph)
    return result


def prolong_phonemes(phoneme_list: List[str], prob: float = PRO_PROB) -> List[str]:
    """
    Duplicate sonorant/vowel phonemes to simulate dysarthric prolongations.
    In real dysarthric speech, vowels and nasals are often stretched.

    Args:
        phoneme_list: Input IPA phoneme tokens.
        prob: Per-sonorant probability of prolongation.

    Returns:
        Phoneme list with duplicated tokens where prolongation was applied.
    """
    result = []
    for ph in phoneme_list:
        result.append(ph)
        key = ph.lstrip("ˈˌ")
        # Prolong if any character in the phoneme is a sonorant/vowel
        if any(c in IPA_SONORANTS for c in key) and random.random() < prob:
            result.append(ph)  # duplication simulates lengthening
    return result


def apply_phoneme_augmentation(
    phoneme_list: List[str],
    do_substitution: bool = True,
    do_deletion: bool = True,
    do_prolongation: bool = True,
) -> List[str]:
    """
    Apply all three phoneme-level augmentations in the standard order:
      1. Substitution (may also delete via empty-string replacement)
      2. Deletion (explicit consonant removal)
      3. Prolongation (sonorant/vowel duplication)

    The order matters: substitution first ensures deletions are coherent,
    prolongation last prevents doubled phonemes from being re-deleted.

    Args:
        phoneme_list: Input IPA tokens.
        do_substitution / do_deletion / do_prolongation: Toggle each op.

    Returns:
        Augmented IPA token list.
    """
    if do_substitution:
        phoneme_list = substitute_phonemes(phoneme_list)
    if do_deletion:
        phoneme_list = delete_phonemes(phoneme_list)
    if do_prolongation:
        phoneme_list = prolong_phonemes(phoneme_list)
    return phoneme_list


# ── IPA back-conversion helpers ───────────────────────────────────────
# Tacotron2 accepts orthographic text, not raw IPA.
# We use a best-effort IPA→English mapping for the augmented tokens
# so the TTS still produces plausible (dysarthric-like) output.

IPA_TO_APPROX_ENGLISH: Dict[str, str] = {
    "p": "p",   "b": "b",   "t": "t",   "d": "d",
    "k": "k",   "ɡ": "g",   "g": "g",
    "f": "f",   "v": "v",   "θ": "th",  "ð": "th",
    "s": "s",   "z": "z",   "ʃ": "sh",  "ʒ": "zh",
    "tʃ": "ch", "dʒ": "j",
    "m": "m",   "n": "n",   "ŋ": "ng",
    "l": "l",   "r": "r",   "w": "w",   "j": "y",
    "h": "h",
    # Vowels (approximate)
    "æ": "a",   "ɑ": "ah",  "ɒ": "o",   "ɔ": "aw",
    "ə": "uh",  "ɛ": "e",   "ɜ": "ur",  "ɪ": "i",
    "ʊ": "oo",  "ʌ": "uh",  "iː": "ee", "uː": "oo",
    "eɪ": "ay", "aɪ": "eye","ɔɪ": "oy", "aʊ": "ow",
    "oʊ": "oh", "ɪər": "ear","ɛər": "air",
    # Simple ASCII vowels pass through
    "a": "a",   "e": "e",   "i": "i",   "o": "o",   "u": "u",
}


def phoneme_list_to_approx_text(phoneme_list: List[str]) -> str:
    """
    Convert an augmented IPA token list to an approximate English
    orthographic string suitable for Tacotron2 TTS input.
    Unknown tokens are passed through unchanged.
    """
    parts = []
    for ph in phoneme_list:
        key = ph.lstrip("ˈˌ")
        parts.append(IPA_TO_APPROX_ENGLISH.get(key, key))
    # Join with spaces so TTS treats tokens individually
    return " ".join(p for p in parts if p)


# =====================================================================
# CELL 7 — TTS synthesizer (singleton, initialised once)
# =====================================================================

class TTSSynthesizer:
    """
    Wraps Coqui TTS (Tacotron2-DDC on LJSpeech).
    Singleton pattern — model is loaded once and reused.

    Why Tacotron2-DDC?
      - No API key required
      - Runs on CPU or GPU
      - ~400 MB download, fast inference
      - Outputs 22050 Hz, resampled to TARGET_SR

    Limitation: accepts orthographic text, not raw IPA.
    The IPA→approximate-text mapping in phoneme_list_to_approx_text()
    bridges this gap. A fully IPA-capable synthesizer (e.g. Piper or
    ESPnet) would be more faithful; see report discussion section.
    """
    _instance: Optional["TTSSynthesizer"] = None

    def __new__(cls):
        if cls._instance is None:
            print("Loading Coqui TTS model (first call only)…")
            cls._instance = super().__new__(cls)
            cls._instance._model = CoquiTTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC"
            )
            print("TTS model ready.")
        return cls._instance

    def synthesize(self, text: str, target_sr: int = TARGET_SR) -> np.ndarray:
        """
        Synthesize speech from text and return a float32 numpy waveform
        resampled to target_sr.

        Args:
            text: Orthographic English text (or phoneme approximation).
            target_sr: Output sample rate in Hz.

        Returns:
            float32 numpy array of shape (n_samples,).
            Returns a quarter-second silence array if text is empty.
        """
        text = text.strip()
        if not text:
            return np.zeros(target_sr // 4, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            self._model.tts_to_file(text=text, file_path=tmp_path)
            wav, _ = librosa.load(tmp_path, sr=target_sr)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return wav.astype(np.float32)


# =====================================================================
# CELL 8 — Phonemizer backend (singleton, initialised once)
# =====================================================================

class PhonemizerBackend:
    """
    Wraps phonemizer's EspeakBackend.
    Singleton — backend is initialised once to avoid repeated overhead.
    """
    _instance: Optional["PhonemizerBackend"] = None

    def __new__(cls):
        if cls._instance is None:
            print("Initialising espeak-ng phonemizer backend…")
            cls._instance = super().__new__(cls)
            cls._instance._backend = EspeakBackend(
                language="en-us",
                preserve_punctuation=False,
                with_stress=True,
            )
            print("Phonemizer ready.")
        return cls._instance

    def text_to_ipa(self, text: str) -> str:
        """
        Convert a plain English word or phrase to IPA.

        Args:
            text: Input English text.

        Returns:
            IPA string (e.g. "ækˈædəmɪk").
        """
        result = self._backend.phonemize([text], njobs=1)
        return result[0].strip() if result else ""


# =====================================================================
# CELL 9 — Full phoneme augmentation pipeline
# =====================================================================

def augment_phoneme_level(
    text: str,
    synthesizer: TTSSynthesizer,
    phonemizer_be: PhonemizerBackend,
    target_sr: int = TARGET_SR,
    do_substitution: bool = True,
    do_deletion: bool = True,
    do_prolongation: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    End-to-end phoneme-level augmentation pipeline:

      Step 1: text → IPA          (espeak-ng)
      Step 2: IPA → token list    (ipa_to_list)
      Step 3: token list → augmented token list
                                  (substitute / delete / prolong)
      Step 4: augmented tokens → approximate English text
                                  (phoneme_list_to_approx_text)
      Step 5: approximate text → waveform
                                  (Coqui TTS Tacotron2-DDC)

    Args:
        text:           Original English transcript.
        synthesizer:    TTSSynthesizer singleton.
        phonemizer_be:  PhonemizerBackend singleton.
        target_sr:      Output sample rate in Hz.
        do_*:           Toggle individual augmentation operations.
        verbose:        Print intermediate IPA steps for debugging.

    Returns:
        float32 numpy waveform at target_sr Hz.
    """
    # Step 1 & 2: text → IPA → token list
    ipa_str      = phonemizer_be.text_to_ipa(text)
    phoneme_list = ipa_to_list(ipa_str)

    if verbose:
        print(f"  Original text : {text}")
        print(f"  IPA           : {ipa_str}")
        print(f"  Tokens        : {phoneme_list}")

    # Step 3: apply augmentations
    aug_list = apply_phoneme_augmentation(
        phoneme_list,
        do_substitution=do_substitution,
        do_deletion=do_deletion,
        do_prolongation=do_prolongation,
    )

    if verbose:
        print(f"  Augmented IPA : {aug_list}")

    # Step 4: convert augmented IPA tokens → approximate English
    approx_text = phoneme_list_to_approx_text(aug_list)

    if verbose:
        print(f"  Approx text   : {approx_text}")

    # Step 5: synthesize waveform
    wav = synthesizer.synthesize(approx_text, target_sr=target_sr)
    return wav


# =====================================================================
# CELL 10 — UASpeech Dataset
# =====================================================================

class UASpeechDataset(Dataset):
    """
    PyTorch Dataset for the Kaggle noise-reduced UASpeech corpus.

    Filename convention (handles both layouts):
      <speaker>/<block>/<word_id>_<mic>.wav
      <speaker>_<block>_<word_id>_<mic>.wav

    Policies:
      "none"     — load raw audio, no augmentation
      "prosodic" — time-stretch + pitch-shift on raw waveform
      "phoneme"  — phoneme-level augmentation via TTS synthesis
    """

    def __init__(
        self,
        root: Path,
        policy: str = "none",
        target_sr: int = TARGET_SR,
        synthesizer: Optional[TTSSynthesizer] = None,
        phonemizer_be: Optional[PhonemizerBackend] = None,
    ):
        self.root          = root
        self.policy        = policy
        self.target_sr     = target_sr
        self.synthesizer   = synthesizer
        self.phonemizer_be = phonemizer_be
        self.manifest      = self._build_manifest()

        if policy == "phoneme" and (synthesizer is None or phonemizer_be is None):
            raise ValueError(
                "policy='phoneme' requires both `synthesizer` and "
                "`phonemizer_be` to be provided."
            )

        print(
            f"[UASpeechDataset] policy={policy!r} | "
            f"{len(self.manifest)} utterances loaded"
        )

    # ── Manifest builder ──────────────────────────────────────────────

    def _parse_word_id(self, stem: str) -> Optional[str]:
        """
        Extract the word ID from a filename stem.

        Handles patterns like:
          CF02_B1_UW1_M7  →  UW1
          UW1_M7          →  UW1
          M05_B2_C13_M5   →  C13
          F02_B1_LK_M5    →  LK
        """
        m = re.search(r'([A-Z]+\d*)_M\d+$', stem, re.IGNORECASE)
        return m.group(1).upper() if m else None

    def _build_manifest(self) -> List[Dict]:
        manifest      = []
        seen_word_ids = set()

        for wav in sorted(self.root.rglob("*.wav")):
            word_id = self._parse_word_id(wav.stem)
            if word_id is None:
                continue

            text = word_id_to_text(word_id)
            if text is None:
                text = word_id.lower()  # fallback: use raw ID as label
            seen_word_ids.add(word_id)

            # Infer speaker from path parts
            speaker = "unknown"
            for part in wav.parts:
                if re.match(r'^[CM][FM]?\d+$', part, re.IGNORECASE):
                    speaker = part
                    break

            manifest.append({
                "path":       wav,
                "transcript": text,
                "word_id":    word_id,
                "speaker":    speaker,
            })

        print(
            f"  Found {len(seen_word_ids)} unique word IDs. "
            f"Sample: {sorted(seen_word_ids)[:8]} …"
        )
        return manifest

    # ── Prosodic augmentation (kept for comparison) ───────────────────

    @staticmethod
    def _augment_prosodic(y: np.ndarray, sr: int) -> np.ndarray:
        if random.random() < 0.5:
            rate = random.uniform(0.85, 1.15)
            y = librosa.effects.time_stretch(y, rate=rate)
        if random.random() < 0.5:
            n_steps = random.uniform(-3, 3)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        return y

    # ── __getitem__ ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict:
        item       = self.manifest[idx]
        transcript = item["transcript"]

        if self.policy == "phoneme":
            # Generate a synthetic dysarthric-like waveform from the transcript.
            # The ORIGINAL audio is NOT used — the augmented audio comes from TTS.
            waveform = augment_phoneme_level(
                text          = transcript,
                synthesizer   = self.synthesizer,
                phonemizer_be = self.phonemizer_be,
                target_sr     = self.target_sr,
            )
            waveform = torch.tensor(waveform, dtype=torch.float32)

        else:
            # Load original audio
            waveform, sr = librosa.load(item["path"], sr=self.target_sr)
            if self.policy == "prosodic":
                waveform = self._augment_prosodic(waveform, sr)
            waveform = torch.tensor(waveform, dtype=torch.float32)

        return {
            "waveform":   waveform,
            "transcript": transcript,
            "word_id":    item["word_id"],
            "speaker":    item["speaker"],
        }


# =====================================================================
# CELL 11 — CTC data collator
# =====================================================================

@dataclass
class DataCollatorCTC:
    """
    Collates variable-length waveforms and transcripts into padded batches
    suitable for HuBERT CTC training.

    The collator:
      1. Encodes waveforms → input_values (float32, normalised)
      2. Encodes transcripts → label token IDs
      3. Masks padding positions with -100 (ignored by CTC loss)
    """
    processor: Wav2Vec2Processor

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        waveforms   = [item["waveform"].numpy() for item in batch]
        transcripts = [item["transcript"]        for item in batch]

        # Audio encoding
        inputs = self.processor(
            waveforms,
            sampling_rate = TARGET_SR,
            return_tensors = "pt",
            padding = True,
        )

        # Label encoding
        with self.processor.as_target_processor():
            label_enc = self.processor(
                transcripts,
                return_tensors = "pt",
                padding = True,
            )

        inputs["labels"] = label_enc.input_ids.masked_fill(
            label_enc.attention_mask.ne(1), -100
        )

        return inputs


# =====================================================================
# CELL 12 — Metrics and train/eval helpers
# =====================================================================

def decode_predictions(
    logits: torch.Tensor,
    processor: Wav2Vec2Processor,
) -> List[str]:
    """Greedy CTC decode: argmax over vocab at each time step."""
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)


def compute_metrics(
    preds: List[str],
    refs: List[str],
) -> Dict[str, float]:
    """Compute WER and CER between predictions and references."""
    return {
        "wer": wer(refs, preds),
        "cer": cer(refs, preds),
    }


def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epoch: int,
) -> float:
    """
    One training epoch with:
      - Mixed precision (AMP) via GradScaler
      - Gradient accumulation (GRAD_ACCUM_STEPS)
      - Gradient clipping (max norm = 1.0)
    """
    model.train()
    total_loss   = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast():
            out  = model(**batch)
            loss = out.loss / GRAD_ACCUM_STEPS  # scale for accumulation

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % 20 == 0:
            alloc = torch.cuda.memory_allocated() / 1e9
            print(
                f"  [Epoch {epoch}] step {step+1}/{len(loader)} | "
                f"loss={loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                f"GPU={alloc:.2f}GB"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    processor: Wav2Vec2Processor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluation loop: computes val loss, WER, and CER.
    Uses greedy CTC decoding.
    """
    model.eval()
    preds_all  = []
    refs_all   = []
    total_loss = 0.0

    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()

        preds = decode_predictions(outputs.logits, processor)

        label_ids = batch["labels"].clone()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(label_ids, group_tokens=False)

        preds_all.extend(preds)
        refs_all.extend(refs)

    metrics        = compute_metrics(preds_all, refs_all)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# =====================================================================
# CELL 13 — Main training run
# =====================================================================

def main():
    # ── Reproducibility ───────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPolicy : {POLICY}")
    print(f"Device : {device}")
    print(f"Model  : {MODEL_NAME}\n")

    # ── Initialise singletons (phoneme policy only) ───────────────────
    synthesizer   = None
    phonemizer_be = None
    if POLICY == "phoneme":
        phonemizer_be = PhonemizerBackend()
        synthesizer   = TTSSynthesizer()

        # Quick sanity check — print augmentation example
        print("\n── Phoneme augmentation sanity check ──")
        test_word = "alphabet"
        test_ipa  = phonemizer_be.text_to_ipa(test_word)
        test_tok  = ipa_to_list(test_ipa)
        test_aug  = apply_phoneme_augmentation(test_tok)
        test_txt  = phoneme_list_to_approx_text(test_aug)
        print(f"  Word    : {test_word}")
        print(f"  IPA     : {test_ipa}")
        print(f"  Tokens  : {test_tok}")
        print(f"  Augment : {test_aug}")
        print(f"  TTS in  : {test_txt}")
        print("────────────────────────────────────────\n")

    # ── Build datasets ────────────────────────────────────────────────
    # Train dataset uses the chosen POLICY.
    # Validation dataset ALWAYS uses "none" — never augment val data.
    train_ds_base = UASpeechDataset(
        root          = DATASET_ROOT,
        policy        = POLICY,
        target_sr     = TARGET_SR,
        synthesizer   = synthesizer,
        phonemizer_be = phonemizer_be,
    )
    val_ds_base = UASpeechDataset(
        root      = DATASET_ROOT,
        policy    = "none",
        target_sr = TARGET_SR,
    )

    n     = len(train_ds_base)
    n_val = max(1, int(n * VAL_SPLIT))
    idx   = list(range(n))
    random.shuffle(idx)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_ds = torch.utils.data.Subset(train_ds_base, train_idx)
    val_ds   = torch.utils.data.Subset(val_ds_base,   val_idx)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # ── GPU memory cleanup ────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    # ── Load processor and model ──────────────────────────────────────
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    model = HubertForCTC.from_pretrained(
        MODEL_NAME,
        mask_time_prob    = 0.0,   # SpecAugment disabled — testing phoneme aug only
        mask_feature_prob = 0.0,
        apply_spec_augment = False,
    ).to(device)

    # Freeze CNN feature extractor to save ~20% GPU memory
    model.hubert.feature_extractor._freeze_parameters()

    # Gradient checkpointing — trades compute for memory
    model.gradient_checkpointing_enable()

    print(
        f"GPU after model load: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated | "
        f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved\n"
    )

    # ── DataLoaders ───────────────────────────────────────────────────
    collator = DataCollatorCTC(processor=processor)

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collator,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collator,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )

    # ── Optimizer, scheduler, AMP scaler ─────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = int(0.1 * total_steps),
        num_training_steps = total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ── Output directory & results log ───────────────────────────────
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    results_file = Path("results_phoneme.csv")
    if not results_file.exists():
        results_file.write_text("policy,epoch,train_loss,val_loss,wer,cer\n")

    best_wer = float("inf")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        val_m = evaluate(model, val_loader, processor, device)

        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"WER={val_m['wer']:.4f} | "
            f"CER={val_m['cer']:.4f}"
        )

        # Log results
        with open(results_file, "a") as f:
            f.write(
                f"{POLICY},{epoch},{train_loss:.4f},"
                f"{val_m['loss']:.4f},{val_m['wer']:.4f},{val_m['cer']:.4f}\n"
            )

        # Save best checkpoint
        if val_m["wer"] < best_wer:
            best_wer = val_m["wer"]
            model.save_pretrained(out / "best_model")
            processor.save_pretrained(out / "best_model")
            print(f"  ✓ New best WER={best_wer:.4f} → saved to {out / 'best_model'}")

    print(f"\nTraining complete.")
    print(f"Best WER : {best_wer:.4f}")
    print(f"Results  : {results_file}")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

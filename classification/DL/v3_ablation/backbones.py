"""
SMILES encoder registry. Every backbone is pinned to an exact Hub revision and
loaded offline, so a Hub update can never change (or break) a result.

All backbones are HF encoders exposing `.encoder.layer` and `.embeddings`, which
is all V3 needs (layer freezing, layer-wise LR decay, multi-layer pooling).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BACKBONES = {
    "molformer": dict(repo="ibm/MoLFormer-XL-both-10pct", revision="7b12d946c181a37f6012b9dc3b002275de070314",
                      remote_code=True, label="MoLFormer-XL (47M, 12L, linear attention)"),
    "chemberta_mlm": dict(repo="DeepChem/ChemBERTa-77M-MLM", revision="ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
                          remote_code=False, label="ChemBERTa-2 77M-MLM (3.4M, 3L)"),
    "chemberta_mtr": dict(repo="DeepChem/ChemBERTa-77M-MTR", revision="66b895cab8adebea0cb59a8effa66b2020f204ca",
                          remote_code=False, label="ChemBERTa-2 77M-MTR (3.4M, 3L)"),
    "chemberta_zinc": dict(repo="seyonec/ChemBERTa-zinc-base-v1", revision="761d6a18cf99db371e0b43baf3e2d21b3e865a20",
                           remote_code=False, label="ChemBERTa zinc-base-v1 (44M, 6L, BPE)"),
}


def load_tokenizer(name="molformer"):
    from transformers import AutoTokenizer
    b = BACKBONES[name]
    return AutoTokenizer.from_pretrained(b["repo"], revision=b["revision"], trust_remote_code=b["remote_code"])


def load_encoder(name="molformer", fast=True):
    """Returns the pretrained encoder (pooler unused). MoLFormer gets the
    sync-free attention patch (bit-identical outputs, see core.patch_molformer_fast)."""
    from transformers import AutoModel
    b = BACKBONES[name]
    enc = AutoModel.from_pretrained(b["repo"], revision=b["revision"], trust_remote_code=b["remote_code"])
    if name == "molformer" and fast:
        from core import patch_molformer_fast
        patch_molformer_fast(enc)
    return enc


def register_remote_code():
    """Makes `transformers_modules.*` importable so notebook pickles can be loaded."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    b = BACKBONES["molformer"]
    get_class_from_dynamic_module("modeling_molformer.MolformerModel", b["repo"], revision=b["revision"])

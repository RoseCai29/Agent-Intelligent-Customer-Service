"""配置加载：读取 config.yaml，并允许环境变量覆盖（KEFU_<大写下划线>）。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_env_overrides(cfg)


def apply_env_overrides(cfg: dict, prefix: str = "KEFU_") -> dict:
    """递归扫描环境变量 KEFU_<PATH> 覆盖配置，PATH 用 _ 连接路径片段。"""
    def walk(node: dict, path: list[str]):
        for key, val in node.items():
            if isinstance(val, dict):
                walk(val, path + [key])
                continue
            env_key = prefix + "_".join(str(p).upper() for p in path + [key])
            if env_key in os.environ:
                raw = os.environ[env_key]
                node[key] = _coerce(raw)
    walk(cfg, [])
    return cfg


def _coerce(raw: str):
    low = raw.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    for probe in (int, float):
        try:
            return probe(raw)
        except ValueError:
            continue
    return raw


def resolve_secret(cfg: dict):
    """从环境变量读取 LLM / embedding 密钥。"""
    llm = cfg["llm"]
    key = os.environ.get(llm.get("api_key_env", "DEEPSEEK_API_KEY"))
    emb = cfg["embedding"]
    emb_key = os.environ.get(emb.get("api_key_env", "SILICONFLOW_API_KEY")) if emb["mode"] == "api" else None
    return {"llm_key": key, "embedding_key": emb_key}
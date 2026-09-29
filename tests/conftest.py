"""Test configuration.

Force the deterministic extractive path so the suite never makes live Gemini calls
(fast, free, and reproducible in CI regardless of any local secrets.toml / env key).
"""
import os

os.environ["MF_DISABLE_GEMINI"] = "1"

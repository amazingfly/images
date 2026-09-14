#!/usr/bin/env python3
"""Securely install and validate the write token needed by the trainer."""

from __future__ import annotations

import getpass
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def main() -> int:
    token = getpass.getpass("Hugging Face write token (input is hidden): ").strip()
    if not token.startswith("hf_"):
        raise RuntimeError("that does not look like a Hugging Face access token")
    who = HfApi(token=token).whoami()
    owner = str(who.get("name", ""))
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    if owner != CONFIG["owner"]:
        raise RuntimeError(f"token belongs to {owner!r}, expected {CONFIG['owner']!r}")
    if role not in {"write", "fineGrained"}:
        raise RuntimeError(f"token role is {role!r}; write access is required")

    destination = Path(CONFIG["token_path"]).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".hf-token-", dir=destination.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    print(f"Installed validated {role} token for {owner} at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

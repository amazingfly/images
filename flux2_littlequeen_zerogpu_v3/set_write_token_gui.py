#!/usr/bin/env python3
"""Install a Hugging Face write token using a desktop password dialog."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
LOG = logging.getLogger("littlequeen-hf-auth")


def dialog(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["zenity", *arguments], capture_output=True, text=True, check=False
    )


def error(message: str) -> None:
    dialog("--error", "--title=Little Queen Hugging Face authentication", f"--text={message}")


def main() -> int:
    while True:
        response = dialog(
            "--password",
            "--title=Little Queen Hugging Face authentication",
            "--text=Paste the NEW amazingfly WRITE token (beginning hf_). It will be validated and stored privately.",
        )
        if response.returncode != 0:
            LOG.warning("authentication dialog canceled")
            return 1
        token = response.stdout.strip()
        if not token.startswith("hf_"):
            message = "That value does not begin with hf_. Paste the generated access token, not your password."
            LOG.warning(message)
            error(message)
            continue
        try:
            who = HfApi(token=token).whoami()
            owner = str(who.get("name", ""))
            role = who.get("auth", {}).get("accessToken", {}).get("role")
        except Exception as exception:
            message = f"Hugging Face rejected the token: {exception}"
            LOG.warning(message)
            error(message)
            continue
        if owner != CONFIG["owner"]:
            message = f"Token belongs to {owner!r}; expected {CONFIG['owner']!r}."
            LOG.warning(message)
            error(message)
            continue
        if role not in {"write", "fineGrained"}:
            message = f"Token role is {role!r}; create a new token with Write permissions."
            LOG.warning(message)
            error(message)
            continue
        break

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
    dialog(
        "--info",
        "--title=Little Queen Hugging Face authentication",
        "--text=Write access validated. The trainer will deploy automatically within one minute.",
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())

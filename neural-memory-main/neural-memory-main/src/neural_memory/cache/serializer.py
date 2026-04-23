"""Serialization for activation cache files.

Supports JSON (fallback) and MessagePack (fast, if available).
Cache files: .neuralmemory/{brain_name}.activation.cache
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.cache.models import ActivationCache

logger = logging.getLogger(__name__)

# Try msgpack for faster serialization
try:
    import msgpack as _msgpack

    _HAS_MSGPACK = True
except ImportError:
    _msgpack = None
    _HAS_MSGPACK = False


def _sanitize_brain_name(brain_name: str) -> str:
    """Sanitize brain name to prevent path traversal.

    Removes path separators, parent directory references, and null bytes.
    """
    sanitized = brain_name.replace("/", "_").replace("\\", "_")
    sanitized = sanitized.replace("..", "_")
    sanitized = sanitized.replace("\x00", "_")
    sanitized = Path(sanitized).name
    return sanitized or "default"


def _get_cache_path(brain_name: str, data_dir: Path | None = None) -> Path:
    """Get cache file path for a brain.

    Args:
        brain_name: Name of the brain
        data_dir: Data directory (defaults to ~/.neuralmemory)

    Returns:
        Path to cache file

    Raises:
        ValueError: If resolved path escapes data directory
    """
    if data_dir is None:
        data_dir = Path.home() / ".neuralmemory"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize brain_name to prevent path traversal
    safe_name = _sanitize_brain_name(brain_name)
    cache_path = (data_dir / f"{safe_name}.activation.cache").resolve()

    # Verify path stays within data_dir (defense in depth)
    if not cache_path.is_relative_to(data_dir.resolve()):
        raise ValueError("Invalid brain name: path traversal detected")

    return cache_path


def save_cache(
    cache: ActivationCache,
    data_dir: Path | None = None,
    use_msgpack: bool = True,
) -> Path:
    """Save activation cache to file.

    Args:
        cache: ActivationCache to save
        data_dir: Data directory (defaults to ~/.neuralmemory)
        use_msgpack: Use msgpack if available (faster)

    Returns:
        Path to saved file
    """
    path = _get_cache_path(cache.brain_name, data_dir)
    data = cache.to_dict()

    if use_msgpack and _HAS_MSGPACK and _msgpack is not None:
        with open(path, "wb") as f:
            _msgpack.pack(data, f)
        logger.debug("Saved activation cache to %s (_msgpack)", path)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("Saved activation cache to %s (json)", path)

    return path


def load_cache(
    brain_name: str,
    data_dir: Path | None = None,
) -> ActivationCache | None:
    """Load activation cache from file.

    Args:
        brain_name: Name of the brain
        data_dir: Data directory (defaults to ~/.neuralmemory)

    Returns:
        ActivationCache if found and valid, None otherwise
    """
    from neural_memory.cache.models import ActivationCache

    path = _get_cache_path(brain_name, data_dir)

    if not path.exists():
        logger.debug("No activation cache found for brain '%s'", brain_name)
        return None

    try:
        # Try msgpack first (binary), fall back to JSON
        if _HAS_MSGPACK:
            try:
                with open(path, "rb") as f:
                    data = _msgpack.unpack(f, raw=False)
                logger.debug("Loaded activation cache from %s (msgpack)", path)
                return ActivationCache.from_dict(data)
            except Exception as e:
                # msgpack can raise various errors for non-msgpack data
                # (UnpackException, ExtraData, UnicodeDecodeError, etc.)
                logger.debug("Msgpack decode failed, trying JSON: %s", e)
                # Fall through to JSON

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded activation cache from %s (json)", path)
        return ActivationCache.from_dict(data)

    except (json.JSONDecodeError, KeyError, ValueError, AttributeError, TypeError) as e:
        logger.warning("Failed to load activation cache: %s", e)
        return None


def delete_cache(
    brain_name: str,
    data_dir: Path | None = None,
) -> bool:
    """Delete activation cache file.

    Args:
        brain_name: Name of the brain
        data_dir: Data directory

    Returns:
        True if deleted, False if not found
    """
    path = _get_cache_path(brain_name, data_dir)
    if path.exists():
        path.unlink()
        logger.debug("Deleted activation cache for brain '%s'", brain_name)
        return True
    return False


def cache_exists(
    brain_name: str,
    data_dir: Path | None = None,
) -> bool:
    """Check if cache file exists."""
    path = _get_cache_path(brain_name, data_dir)
    return path.exists()

"""Wrap stage: package training outputs into a zip archive."""
import logging
import zipfile
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Zip the output directory into output.zip."""
    output_dir = Path(cfg.paths.output_dir)

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    files = list(output_dir.rglob("*"))
    if not any(f.is_file() for f in files):
        raise FileNotFoundError(f"Output directory is empty: {output_dir}")

    archive_path = Path("output.zip")
    log.info("Packaging %s into %s", output_dir, archive_path)

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(output_dir.parent))

    log.info("Archive created: %s (%.1f MB)", archive_path, archive_path.stat().st_size / 1e6)

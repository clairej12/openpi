#!/usr/bin/env python3
"""Shared dataset helpers for DROID/LeRobot scripts."""

import glob
import os
import re


def discover_lerobot_parquets(data_root: str):
    """Yield (chunk_idx, file_idx, parquet_path) for LeRobot parquet shards."""
    pattern = os.path.join(data_root, "data", "chunk-*", "file-*.parquet")
    for path in sorted(glob.glob(pattern)):
        m_chunk = re.search(r"chunk-(\d+)", path)
        m_file = re.search(r"file-(\d+)\.parquet$", path)
        if not m_chunk or not m_file:
            continue
        yield int(m_chunk.group(1)), int(m_file.group(1)), path

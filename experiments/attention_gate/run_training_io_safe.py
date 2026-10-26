"""Explicit operational entry point for the unchanged v1_2 scientific runner."""
import json
import sys
from pathlib import Path
import common_runner_v2 as fast
import safe_progress_io as io


if __name__ == '__main__':
    io.install(fast)
    try:
        fast.main()
    finally:
        # Scientific output folder is created only by the frozen runner.
        if '--output' in sys.argv:
            folder = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
            if folder.is_relative_to(fast.core.HERE) and folder.is_dir():
                fast.core.json_write(folder / 'io_health.json', {'adapter': 'safe_progress_io_v1',
                    'adapter_sha256': fast.core.digest(Path(io.__file__)), 'counters': io.STATS,
                    'scientific_settings_and_sources_unchanged': True})

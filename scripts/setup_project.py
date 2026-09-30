"""Create a portable environment, optionally restore all release assets, and verify the demo.

Windows: py -3.11 scripts/setup_project.py --full-data --run
Linux/macOS: python3.11 scripts/setup_project.py --full-data --run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data", action="store_true", help="Restore dataset and research release assets")
    parser.add_argument("--skip-install", action="store_true", help="Use an already installed .venv")
    parser.add_argument("--run", action="store_true", help="Start Streamlit after verification")
    args = parser.parse_args()
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise SystemExit("Use Python 3.11 or 3.12, the versions verified for this project.")
    env_dir = ROOT / ".venv"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    temp = ROOT / "runtime" / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp))
    # ensurepip invoked by venv also inherits a temp directory on the project drive.
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(temp)
    if not python.exists():
        if args.skip_install:
            raise SystemExit("No .venv found. Omit --skip-install on first setup.")
        print(f"Creating environment: {env_dir}", flush=True)
        venv.EnvBuilder(with_pip=True).create(env_dir)
    def run(*arguments: str) -> None:
        subprocess.run([str(python), *arguments], cwd=ROOT, env=env, check=True)
    run("-c", "import sys; assert sys.version_info[:2] in ((3,11),(3,12)), 'Existing .venv uses an unsupported Python'")
    if not args.skip_install:
        run("-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt")
    if args.full_data:
        run("scripts/restore_assets.py", "--include-research", "--prepare")
    run("scripts/smoke_check.py")
    print("\nReady. Open http://127.0.0.1:8501 after starting the server, then click Load demo.", flush=True)
    print(f'Run again: "{python}" -m streamlit run app/streamlit_app.py --server.address 127.0.0.1', flush=True)
    if args.run:
        run("-m", "streamlit", "run", "app/streamlit_app.py", "--server.address", "127.0.0.1")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Setup stopped because a command failed (exit {exc.returncode}). See the error above.") from exc

# Run DRISHTI on a new laptop

The app, saved models, synthetic demo, public data, evaluation results and tests are in Git.
The **complete original dataset** and historical research are in the private GitHub release
[`migration-v1-2026-09-30`](https://github.com/Hopper543/DRISHTI/releases/tag/migration-v1-2026-09-30).
Use the same GitHub account, or another account with access. No API key is needed for the model.
Allow at least 4 GB free on the drive where you clone the project for the environment, archives and extraction.

## 1. Install prerequisites and clone

Install Python **3.11** (recommended; 3.12 also tested), Git and GitHub CLI. The application does not
depend on any external AI API.

```powershell
gh auth login
gh repo clone Hopper543/DRISHTI
cd DRISHTI
```

These clone commands also work in Linux/macOS shells. The repository's default branch is
`feature/prototype-v1`; do not assume it is called `main`. Cloning checks out the default automatically.

## 2. Restore everything and start

Windows PowerShell:

```powershell
py -3.11 scripts/setup_project.py --full-data --run
```

Linux/macOS:

```bash
python3.11 scripts/setup_project.py --full-data --run
```

Use `py -3.12` / `python3.12` if you installed Python 3.12. No shell activation or PowerShell
execution-policy change is required. The helper creates `.venv`, installs `requirements.txt`,
downloads both release archives, verifies their exact hashes, extracts them, restores only the
local-only app tables, and checks the shipped demo before starting Streamlit.
Open **http://127.0.0.1:8501** and click **Load demo**. Expected: **147 devices, 50 PASS, 1 FAIL,
96 ESCALATE**. This is a local address, not a public demo link.

On Linux, if LightGBM reports missing `libgomp.so.1`, install your distribution's OpenMP runtime
(Debian/Ubuntu: `sudo apt-get install libgomp1`). On macOS, if LightGBM reports missing `libomp`,
install the OpenMP runtime (`brew install libomp` when Homebrew is installed).
Do not change model versions or retrain to fix an environment error.

For a quick demo without the large archive or GitHub CLI, omit `--full-data`. All main pages and
models work with the committed demo/public data; the real explorer then uses the public subset.

## 3. Start again later

Windows:

```powershell
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Linux/macOS:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Stop the server with Ctrl+C. No model training is needed to start the app.

## Cloning versus forking

Moving to another laptop with your existing `Hopper543` account requires a **clone**, not a fork.
If a different authorised account needs its own copy, it may fork through GitHub if the private
repository's policies allow it:

```bash
gh repo fork Hopper543/DRISHTI --clone
cd DRISHTI
```

Release attachments are downloaded explicitly from `Hopper543/DRISHTI` in the manifest. A fork
does not make those original private assets accessible to someone without upstream read access.
If transferring ownership/access permanently, copy the release assets to the destination private
repository and update the manifest's repository field, preserving the checksums. Do not assume
that a Git clone alone includes the full dataset ZIP.

## Manual restore / offline transfer

With an installed virtual environment:

```powershell
.venv\Scripts\python.exe scripts/restore_assets.py --include-research --prepare
```

The helper downloads the pinned release, checks SHA-256, checks ZIP paths and extracts:

| Location | Contents |
|---|---|
| `runtime/downloads/` | Verified downloaded ZIPs |
| `runtime/datasets/drishti_dataset_v1/` | 68 files: full CSV/Parquet data, raw PDFs/MAT/IGBT archive, source terms, templates, dataset scripts/tests, original benchmark results |
| `runtime/datasets/drishti_research_context/` | 101 files: historical scripts, source snapshots, OCR/extraction and PPT-review material |
| `data/local/` | Full real measurements/drift view, folds and quarantined IGBT scalar table used by local data loaders |

For offline restoration, obtain both release ZIPs on an authenticated computer, transfer them,
then run (replace the example directory):

```powershell
.venv\Scripts\python.exe scripts/restore_assets.py --archive-dir "D:\TransferredArchives" --include-research --prepare
```

Use `.venv/bin/python` on Linux/macOS. Offline restoration needs no GitHub CLI, but dependencies must
already be installed. Existing extracted files are verified; edited files are not silently overwritten.
Use `--destination` with a new directory if retaining an edited extraction. Archive integrity is
pinned by `data/dataset_release.json`, including the original dataset SHA-256
`5db4d0d69856c18af160036d94d5a53b048f96a1e4509b6ba208ebc328757997`.

## Verification and reproducibility

```powershell
.venv\Scripts\python.exe scripts/smoke_check.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/check_repo.py
.venv\Scripts\python.exe scripts/generate_synthetic.py --check
.venv\Scripts\python.exe runtime/datasets/drishti_dataset_v1/scripts/verify_hashes.py
```

See README.md for optional training/evaluation commands. The original dataset package contains its own
requirements and benchmark; do not confuse its original evaluation folder with the app's current
`evaluation_results/`. Keep restored delivery files unchanged if you want their hashes to continue matching.

## Scope of the backup

The backup includes all currently collected dataset files, including quarantined IGBT material,
not merely the public subset. Optional MOSFET #13 and capacitor #12 archives were never downloaded;
the original package includes their optional downloader, not their contents.
The PPT, historical idea brief, feature guide, current source, models, tests, reports and screenshots
are committed. Environments, installed libraries, credentials and transient runtime logs are excluded.
Research scripts with old machine paths are preserved as historical context, not supported setup commands.

This repository and its full-data release remain **private**. Several sources have no explicit
redistribution licence recorded; private backup does not grant new reuse rights. Review source terms
before making the repository or release public. Scientific caveats and IGBT quarantine remain in force.

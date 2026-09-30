# Deploying the public demo (Streamlit Community Cloud)

The repository stays **private**; only the dashboard is public. Streamlit Community Cloud reads the repo with the
owner's GitHub permission and serves `app/streamlit_app.py`.

## One-time setup (repo owner, ~5 minutes)

1. Open https://share.streamlit.io and choose **Continue with GitHub** (account `Hopper543`). Approve Streamlit's
   access to the private `DRISHTI` repository when GitHub asks.
2. Click **Create app** → **Deploy a public app from GitHub** (wording may vary slightly).
3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `Hopper543/DRISHTI` |
   | Branch | `feature/prototype-v1` (or `main` after merging) |
   | Main file path | `app/streamlit_app.py` |
   | App URL | e.g. `drishti-agamemnon` → `https://drishti-agamemnon.streamlit.app` |
   | Advanced settings → Python version | **3.11** (tested; 3.12 also passes CI) |
4. Click **Deploy**. The first build installs `requirements.txt` (plus `libgomp1` from `packages.txt` for
   LightGBM) and takes a few minutes.
5. In the app's **Settings → Sharing**, make sure it is viewable by **anyone** (apps from private repos can
   default to invited viewers only). Check the link in a private/incognito window, signed out.

No secrets, API keys or external services are needed; models and demo data are in the repository.

## Before a presentation

* Free-tier apps go to sleep after a period without visitors. Open the link 5–10 minutes beforehand; a sleeping
  app shows a "wake up" button and takes a minute or two to start.
* Pushing to the deployed branch redeploys the app automatically.

## Behaviour on the public instance

* Everyone shares one server: the **audit history** lists all visitors' demo runs and is reset whenever the app
  restarts (the cloud file system is not persistent). It contains run IDs, hashes and decision counts, no
  personal data.
* Uploads are processed in memory for the session; nothing is stored except the audit entry.
* The public instance only has the committed data: synthetic fixtures, SECOM and the redistributable NASA
  subsets (AD620, ReRAM, OP484, U309). NDS352, AD648, capacitor #14 and IGBT stay local.

## Putting the link in the PPT

Select the text or button on the slide → **Insert → Link** (Ctrl+K) → paste the `https://…streamlit.app` URL.
In slideshow mode, clicking it opens the site in the default browser. Also put the URL as visible text (or a QR
code) so judges viewing a PDF export can type it.

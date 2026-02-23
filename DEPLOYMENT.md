# Streamlit Intranet Deployment (Windows)

This repo is a Streamlit multi-page dashboard. Entry point: `Home.py`.

## 1) Recommended intranet architecture

- Run **one** Streamlit instance on an internal Windows Server/VM.
- Users access it via either:
  - Direct: `http://SERVERNAME:8501` (simplest)
  - Reverse proxy (recommended for TLS + Windows/AD auth): `https://intranet.company.local/me-dashboard`

Why one instance: this app writes to CSV and SQLite under `data/` (e.g. `data/BreakdownReport.csv`, `data/workshop.db`, `data/asset_log.db`). Multiple instances writing to the same files can cause conflicts.

## 2) Server prerequisites

- Windows Server (or always-on Windows PC) on the intranet
- Python 3.10+ installed system-wide
- Outbound access to your internal PyPI mirror (or allow `pip` to install packages)
- An always-on folder location for the app (local disk recommended)
- Permissions for the service account to read/write:
  - `data/` (CSV + SQLite DB files)
  - `images/` (uploaded asset images)

## 3) First-time setup (one-time)

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick smoke run (interactive):

```powershell
streamlit run Home.py
```

Then browse from the server itself: `http://localhost:8501`.

## 4) Make it reachable on the intranet

### Option A: Direct access (no reverse proxy)

1. Open the inbound port on Windows Firewall (example 8501).
2. Ensure `.streamlit/config.toml` has:
   - `server.address = "0.0.0.0"`
   - `server.port = 8501`

Users browse: `http://SERVERNAME:8501`.

### Option B: Reverse proxy (recommended)

Put Streamlit behind IIS (ARR + URL Rewrite) or Nginx to get:

- HTTPS/TLS termination with a company certificate
- Central authentication (Windows/AD, SSO, or your IdP)
- Friendly URL / sub-path routing

#### Base URL path

If you reverse-proxy under a sub-path (example `/me-dashboard`), set:

```toml
# .streamlit/config.toml
[server]
baseUrlPath = "me-dashboard"
```

If you proxy at the domain root, you do not need `baseUrlPath`.

#### Authentication

Streamlit itself is not an enterprise auth gateway. In intranet deployments, authenticate at the reverse proxy layer:

- IIS: Windows Authentication (NTLM/Kerberos) or your SSO module
- Nginx: auth_request / SSO integration

## 5) Run as a background service (Windows)

### Option A: Windows Task Scheduler (simple)

Create a task that runs at startup under a service account.

Program/script:
- `powershell.exe`

Arguments:

```powershell
-NoProfile -ExecutionPolicy Bypass -Command "cd 'D:\Script\Dashboard Apps'; .\.venv\Scripts\Activate.ps1; streamlit run Home.py --server.headless true"
```

Set:
- Run whether user is logged on or not
- Start in: your repo folder
- Restart on failure (recommended)

### Option B: NSSM (service wrapper)

If your company allows it, NSSM can run Streamlit as a Windows service.

- Application: `D:\Script\Dashboard Apps\.venv\Scripts\streamlit.exe`
- Arguments: `run Home.py --server.headless true`
- Startup directory: `D:\Script\Dashboard Apps`

## 6) Data, backups, and reliability

- The app persists state under `data/`:
  - Assets CSV: `data/DataBase_ME_Asset.csv`
  - Task reports CSV: `data/BreakdownReport.csv`
  - Workshop DB: `data/workshop.db`
  - Logs DB: `data/asset_log.db`
  - Verification DB: `data/regdata.db`
- Back up `data/` and `images/` (daily scheduled copy is typical).
- SQLite is fine for light/moderate intranet usage, but avoid placing DB files on flaky network shares.

## 7) Updating the app

1. Stop the scheduled task/service.
2. Pull/copy the new code.
3. Re-activate venv and update deps if needed:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Start the service again.

## 8) Troubleshooting

- Port in use: change `server.port` in `.streamlit/config.toml`.
- Can’t access from other PCs: check Windows Firewall + network ACLs.
- Reverse proxy under a sub-path breaks links: set `server.baseUrlPath`.
- File permission errors: ensure the service account has Modify rights on `data/` and `images/`.

## 9) Can this be an EXE?

Yes, but with an important caveat: a Streamlit app is a **web app**. Even if you wrap it in an `.exe`, it will still start a **local server process** and open a browser tab (or you browse to `http://localhost:8501`).

### Best practice (intranet)

For multiple users, prefer the **server deployment** described above (one always-on instance on a Windows Server/VM). That avoids every user running their own copy and writing to CSV/SQLite in `data/`.

### If you still want an `.exe` (single-PC / demo)

There are two common approaches:

1) **Launcher EXE (recommended if you must use EXE)**
  - Build a small executable that runs `streamlit run Home.py` from the repo folder.
  - You still ship Python + dependencies (or require Python installed).
  - This keeps `data/` and `images/` as normal folders next to your code, so your CSV/SQLite writes behave normally.

2) **Bundle everything into one EXE (not recommended for this app)**
  - Tools like PyInstaller can produce a single-file executable, but apps that read/write local data like `data/*.csv` and `*.db` are easy to break because “onefile” apps typically extract to a temporary folder at runtime.
  - If you go this route, you usually need extra code changes to control where persistent files live.

If you tell me which outcome you want (A) “EXE that launches on one PC” or (B) “central intranet server for many users”, I can write the exact run script / packaging config for your choice.

# Deploying (free)

## Recommended — Streamlit Community Cloud

The simplest free host for this app. It installs dependencies with **uv** natively when a
`uv.lock` is present (the highest-priority dependency file), so the deploy stays fully
uv-based — no `requirements.txt` required.

**One-time prep (local):**
```bash
uv lock                      # generate uv.lock from pyproject.toml
git add uv.lock
git commit -m "chore: add uv.lock for reproducible deploys"
git push
```

**Deploy:**
1. Push the repo to **public** GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → choose your repo + branch, main file `app.py`.
3. **Advanced settings** → set **Python 3.12** (match `.python-version`) and paste your secrets:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
   (Same shape as [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) — never commit the real key.)
4. Deploy. Community Cloud reads `uv.lock` and installs with uv.

**No-key demo:** deploy **without** setting `GEMINI_API_KEY` and the app defaults to **offline
replay** (the sidebar toggle defaults on) — the dashboard and per-snippet live demo populate
from the dataset's reference solutions. Add the key later to switch to real models.

**Note:** free apps **sleep after inactivity** — keep the 2 screenshots + a short Loom as a
fallback, and the offline replay means a cold reviewer still sees it work.

## Alternative — Hugging Face Spaces

Also free and easy: create a **Streamlit** Space, push the repo, and add `GEMINI_API_KEY`
under the Space's **Settings → Variables and secrets**. Spaces support uv as well. Comparable
effort; pick whichever account you already have.

---

Sources: [Streamlit Community Cloud — app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[uv on Community Cloud (discussion)](https://discuss.streamlit.io/t/install-dependencies-in-streamlit-cloud-based-on-uv-pyproject-toml/79557).

# Apply this to your Retention-radar repo (about 10 minutes)

1. Copy everything in this folder into the ROOT of your local Retention-radar repo, overwriting when asked.
   (Windows: make sure the hidden `.github` folder is copied too; this is the folder your first upload skipped.)
2. Remove the committed cache files:
       git rm -r --cached churn/__pycache__
3. Retrain with the pinned library version so artifacts match the README:
       python -m venv .venv && .venv\Scripts\activate        (macOS/Linux: source .venv/bin/activate)
       pip install -r requirements-dev.txt
       python -m churn.train
       python -m churn.fairness
       python -m pytest -q          # expect 12 passed
   Expected: hold-out AUC 0.8382, PR-AUC 0.6813, threshold 0.15, profit 19,680.
   If your numbers differ, run `python -m churn.report` and paste its output over the README results section.
4. Optional MLflow: pip install -r requirements-mlflow.txt, then python -m churn.train --mlflow, then mlflow ui
5. Commit and push (use the command line, NOT drag-and-drop, so .github is included):
       git add -A && git commit -m "Restore API, add CI, MLflow, fairness audit, deploy config" && git push
6. Deploy: create a blank Docker Space on huggingface.co, then
       bash deploy/huggingface/push.sh <your-hf-username>/retention-radar
   Put the live URL in the README "Live demo" line and in your resumes.

# run-001 — Commentary

Narrative observations from each monitoring tick — interpretation, concerns, notable trends.

---

### 09:11 PDT — FAILED

run-001 failed at `baseline_eval` — never reached any actual inference. The `24.12-py3` PyTorch image (2.5.x) is incompatible with the latest `transformers` pip-installed into the container. Specifically, `transformers` 4.47+ imports `TransformGetItemToIndex` from `torch._dynamo` which doesn't exist until PyTorch 2.6. The 26.04-py3 image (PyTorch 2.6, already cached at 11 GB) would fix this cleanly; alternatively, pinning `transformers<4.47` keeps the older image but risks other incompatibilities down the line. Switching to 26.04-py3 is the cleaner fix and avoids another cold pull.

### 09:05 PDT

The baseline_eval pod has been PodInitializing for over 6 minutes, blocked on pulling `nvcr.io/nvidia/pytorch:24.12-py3`. The 26.04-py3 tag was already cached (11 GB) but 24.12-py3 is not — this is a cold pull. Expect it to take 10-20 more minutes depending on NGC download speed. Once the pull completes, pip installs (transformers, scikit-learn, mlflow, kfp) will add another 1-2 minutes before actual eval starts. Also noticed the pod resources embed only `{"limits": {"memory": "32G"}}` — no GPU limit visible in the inline spec. This could mean the GPU request was dropped during compilation, or it's set via a separate resource-patch path; worth verifying once the pod goes Running. Baseline eval with CPU fallback would still work but take longer.

### 08:59 PDT

The first two stages (download_model and prepare_dataset) completed fast — under 3 minutes total. The pipeline is now entering baseline_eval, where DNABERT-2 is loaded with a randomly initialized classification head and evaluated on the val set. Accuracy at this point is expected to be ~50% and AUC ~0.50, since the head has random weights — this is the correct and expected baseline for a classification fine-tuning pipeline. No MLflow entries yet, which is normal since eval hasn't started. Watch for MLflow `baseline_accuracy` and `baseline_auc` metrics to appear once the pod transitions to Running and eval begins.

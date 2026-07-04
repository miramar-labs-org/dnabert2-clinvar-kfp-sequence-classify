# run-017 — Commentary

Narrative observations from each monitoring tick — interpretation, concerns, notable trends.

---

### 14:36 PDT

Run-017 completed end to end and the workflow is Succeeded. The deployment gate passed with accuracy_delta +0.2384 and postft_auc 0.9136 against a 0.70 AUC threshold. This run validates both parts of the fix: the bundled DNABERT flash-attention path was bypassed during baseline/fine-tune, and the saved fine-tuned artifact could be reloaded in a separate post_finetune_eval component. The final model result is substantially better than baseline on held-out test data.

### 14:34 PDT

Post-finetune evaluation succeeded and the result is strong: accuracy 0.8364 and AUC 0.9136 on 4,885 held-out test samples. Compared with the baseline_eval sample metrics of accuracy 0.598 and AUC 0.5009, this is a clear lift. Operationally, the saved custom-code DNABERT artifact reloaded successfully in a fresh component, which was the main risk after fine-tune. The run has moved into the gate/summary tail of the pipeline.

### 14:31 PDT

Fine-tune completed cleanly and handed off to post_finetune_eval. The final epoch eval_loss was 0.4162, which is slightly worse than the epoch-2 eval_loss of 0.4051, so the best validation point remains epoch 2. The important operational result is that the model artifact saved and uploaded successfully, including the custom DNABERT code files. The post_finetune_eval pod is now running and should test whether that saved artifact reloads correctly in a fresh component.

### 14:30 PDT

The training loop has reached 3666/3666 and the pod is now in final evaluation/save territory. MLflow has not yet recorded the final eval metrics; its latest training tick is step 3660 with loss 0.2628 and the previous best eval_loss remains 0.4051 from epoch 2. The active pod is still Running, which is expected while final eval and artifact writes complete. The next check should confirm whether final eval improved further and whether KFP successfully starts post_finetune_eval.

### 14:26 PDT

Fine_tune is effectively at the finish line: step 3500/3666, about 95.5% complete. The latest loss is 0.2342 and the learning rate is nearly decayed out at 9.11e-07, with GPU utilization still at 96%. The next stage should be final evaluation/model save, then KFP should launch post_finetune_eval. This is the transition point most worth watching because it exercises the saved model artifact rather than the already-loaded training process.

### 14:21 PDT

Fine-tuning is in the final stretch at step 3190/3666, with roughly nine minutes of training left at the current rate. The latest training loss is 0.2573, lower than prior ticks, and there are still no errors on the active `cdbj8` prefix. The second epoch eval_loss of 0.4051 remains the best validation signal so far. The next check should either catch late training near completion or the handoff into final epoch evaluation/model save.

### 14:15 PDT

The second epoch evaluation landed and improved meaningfully: eval_loss dropped from 0.4441 after epoch 1 to 0.4051 after epoch 2. Training loss is also lower at 0.3029, and the run is now about 79% complete. This is the strongest signal so far that the fine-tune is doing useful work rather than only memorizing noise. The remaining risk is no longer training stability; it is the transition into post_finetune_eval and whether the saved model artifact includes everything needed for custom-code reload.

### 14:05 PDT

The run is now well into the second half of training at step 2350/3666. Loss is down to 0.3122 and the learning rate has decayed to 7.18e-06, with no pod errors on the active `cdbj8` prefix. The next useful signal should be the second epoch evaluation around step 2444; if eval_loss improves from 0.4441, that would support the training trend seen so far. Training should finish in roughly 25 minutes, after which the main risk shifts to whether post_finetune_eval can reload the fine-tuned DNABERT custom-code artifact cleanly.

### 13:54 PDT

Fine-tuning is still clean and is now almost halfway through at step 1770/3666. The training loss has moved down to 0.3450, and the first epoch evaluation produced eval_loss=0.4441, so the model is learning without obvious instability. GPU utilization remains high at 96%, with VRAM up to about 20.6 GiB after checkpoint/eval activity. The next thing to watch is whether the loss keeps improving through the second epoch and whether post_finetune_eval can load the saved custom-code model once training completes.

### 13:33 PDT

Fine-tuning is now materially underway at step 680/3666 (~18.5%) with the active `cdbj8` fine_tune pod still Running. The loss has dropped to 0.4912 by MLflow's latest tick, which is a good early sign given the random-head baseline and class imbalance. GPU SM utilization is high at 96% with about 18.3 GiB VRAM in use, so the job is compute-bound and making steady progress. At the current ~1.09 seconds/step rate, training has roughly an hour left before post_finetune_eval starts.

### 13:20 PDT

The `flash_attn_qkvpacked_func = None` fix worked — baseline_eval completed successfully for the first time. The baseline accuracy is 0.598 (59.8%) with AUC of 0.5009 (essentially random). The 59.8% accuracy on a randomly-initialized head makes sense for a 78%/22% class-imbalanced dataset where the model's attention patterns are effectively random — it's somewhere between always-predict-majority (78%) and pure-chance (50%), likely because with random weights the model sometimes favors one class and sometimes the other. AUC ~0.50 confirms pure chance, as expected. Now fine_tune is running with 39K training samples, 3,666 steps total on the full 117M parameter model. The key question: will fine-tuning on 39K variant sequences teach DNABERT-2 to distinguish pathogenic from benign variants?

### 13:17 PDT

Run-017 applies the correct fix for the flash_attn Triton API incompatibility. The previous attempt (run-016) blocked the `flash_attn` pip package via sys.modules, but DNABERT-2 uses a BUNDLED `flash_attn_triton.py` that it ships as part of the model — an entirely different import path. The bundled file imports successfully (it's valid Python), so the Triton JIT compilation error only hits at forward-pass time when `flash_attn_qkvpacked_func(qkv, bias)` is invoked. The correct fix: after `from_config()` loads bert_layers.py into sys.modules, override `flash_attn_qkvpacked_func = None` on that module. bert_layers.py lines 125 and 161 explicitly check this variable before using flash attention — if None, standard PyTorch attention is used. The key milestone to watch for this run: baseline_eval should complete without the Triton error and report ~78% accuracy (random head predicting the majority class "Benign").

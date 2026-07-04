# run-016 — Commentary

Narrative observations from each monitoring tick — interpretation, concerns, notable trends.

---

### 13:03 PDT

Run-016 fixes the flash_attn Triton compilation error that killed run-015. DNABERT-2's `BertUnpadSelfAttention` imports `flash_attn` and uses its Triton JIT kernel for attention computation, but the installed Triton removed the `trans_b` kwarg from `tl.dot()`. By setting `sys.modules['flash_attn'] = None` before `from_config()`, the `try/except ImportError` in bert_layers.py will catch the blocked import and fall back to standard PyTorch attention. The from_config + manual weight loading fix from run-014 is still in place and confirmed working. The key milestone this run: baseline_eval should complete inference without the Triton error and report ~50% accuracy (random head).


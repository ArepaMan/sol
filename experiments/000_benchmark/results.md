# 000 — VRAM / throughput benchmark

Device: NVIDIA GeForce RTX 4070 Laptop GPU | 8.0 GiB | sm_89 | bf16=True

| batch_size | grad_checkpointing | compile | peak VRAM (MiB) | tokens/s | status |
|---|---|---|---|---|---|
| 4 | True | False | 1460 | 22,484 | ok |
| 4 | False | False | 2137 | 28,722 | ok |
| 8 | True | False | 2247 | 23,110 | ok |
| 4 | True | True | — | — | BackendCompilerFailed: backend='inductor' raised: |
| 16 | False | False | 6337 | 30,262 | ok |
| 32 | False | False | 11965 | 4,934 | ⚠️ shared-memory spill |
| 16 | True | False | 3836 | 22,901 | ok |
| 32 | True | False | 7016 | 22,139 | ok |
| 64 | False | False | — | — | OOM |

### Failure details

**b4 ckpt=True compile=True:**
```
BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at https://github.com/openai/triton

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

```
**b64 ckpt=False compile=False:**
```
OOM
```

**Chosen config** (batch_size=4, gradient_checkpointing=False, compile=False): peak VRAM 2137 MiB (under the 7400 MiB target), 28,722 tokens/s.
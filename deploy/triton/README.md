# Triton model repository (B14-T04)

```
deploy/triton/models/
└── head_a/
    ├── config.pbtxt      # ONNX Runtime backend, dynamic batching, reject-on-timeout
    └── 1/model.onnx      # not committed — produced by the export step below
```

## Produce the model

```bash
pip install onnx onnxruntime
python -m ml.export.distill --config ml/training/configs/distill_head_a.yaml
python -m ml.export.to_onnx --checkpoint runs/distilled/head_a_student.pt --out deploy/triton/models/head_a/1/model.onnx
python -m ml.export.parity_check --checkpoint runs/distilled/head_a_student.pt --onnx deploy/triton/models/head_a/1/model.onnx
```

## Run Triton and point the gateway at it

```bash
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v "$PWD/deploy/triton/models:/models" nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
```

Then set `VG_INFERENCE_BACKEND=triton` and `VG_TRITON_URL=localhost:8000` for the gateway.
`services/inference/backends.py:TritonBackend` sends batches over HTTP (`pip install tritonclient[http]`).

For a CPU-only deployment, delete the `optimization` block and set `kind: KIND_CPU`.

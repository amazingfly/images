# Workflow support index

`workflows.json` is the machine-checked source for this index. Supported means
maintained entry point with CPU checks, not a claim of current GPU availability
or deployment. Archived source stays in place to preserve historical paths.

| Workflow | Status | Requirements |
| --- | --- | --- |
| `sd15` | supported | SD15 runtime/weights |
| `dataset-curation` | supported | Image runtime and vision server |
| `sdxl-backend` | supported | SDXL tool and weights |
| `accessory-backends` | supported | Torch/Transformers, accessory resources |
| `flux-training-v3` | experimental | Versioned training experiment; deployment is external state |
| `flux-training-v1-v2` | archived | Historical recipes; retained service compatibility |
| `flux-evaluation` | experimental | Model-specific evaluation |
| `sa3-image-copy` | archived | Reference only |

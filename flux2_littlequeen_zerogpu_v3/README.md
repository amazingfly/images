# Little Queen v3

See [STRATEGY.md](STRATEGY.md) for dataset choices, identity prefix and evaluation.
The deployed Space retains the name `amazingfly/little-queen-flux2-klein-trainer-v2`
because the account already has two ZeroGPU Spaces. Its dataset and output point
to separate v3 repositories; the v2 model remains saved in its original repository.

Prepare: `python3 flux2_littlequeen_zerogpu_v3/prepare_identity_v3.py`
Deploy: `python3 flux2_littlequeen_zerogpu_v3/deploy.py`
Start/resume: `systemctl --user start littlequeen-flux2-v3.service`
Status: `python3 flux2_littlequeen_zerogpu_v3/status.py`
Logs: `tail -f sdxl_littlequeen_v1/flux2_klein_training_v3/training.log`

The service is enabled to resume after login/reboot. The local computer must be
running for it to request further GPU segments; Hub checkpoints survive downtime.
The proven v2 supervisor, deployment and official-trainer patch are reused.
The inherited `dataset_spec.json` and `test_package.py` describe the previous
dataset builder; the authoritative v3 data is built by `prepare_identity_v3.py`.

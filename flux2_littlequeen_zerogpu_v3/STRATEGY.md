# Little Queen FLUX.2 Klein v3

V3 is a fresh adapter on FLUX.2-klein-base-4B. It does not initialize from v2.
The source data remains SDXL-derived: 20 visually selected originals from v2's
30-image dataset, plus eight manually checked portrait crops. Crops emphasize
existing facial information; they are not new independent images or added detail.

The selection removes the most divergent faces and proportions. Original images
retain body proportions and varied poses. Face crops increase facial coverage in
training. This still has a limited expression range and strong garden, crown and
pink-dress correlations. It cannot establish generalization by itself.

Training and inference share this identity prefix:

> LQK4N, a little queen with a rounded childlike face, large warm brown eyes,
> a small nose, soft round cheeks, and long chestnut-brown hair with bangs

Append scene/action and desired clothing. Attach the trigger to the queen in
multi-character prompts, and describe other characters separately. Do not place
the trigger as a detached style label. Descriptions are support, not proof of
identity consistency. Test both with and without the identity description.

Rank/alpha 32, learning rate 7e-5, BF16, 1024 aspect-ratio buckets, 1400 steps.
Use conservative 160-step segments and the established quota-aware supervisor.
Every completed segment retains an inference adapter for checkpoint comparison.
Full optimizer state is uploaded atomically with progress after every segment.

Before promoting v3, compare early/middle/final checkpoints against v2 at matched
seeds on portraits, profiles, expressions, full-body action, plain backgrounds,
different clothes, and an adult witch beside the queen. Check face, proportions,
identity leakage, duplicate people, lettering and anatomy. The old three evaluation
scenes are retained as external test prompts, not used as training images.

Run: `bash flux2_littlequeen_zerogpu_v3/start.sh`
Status: `python3 flux2_littlequeen_zerogpu_v3/status.py`
Logs: `sdxl_littlequeen_v1/flux2_klein_training_v3/training.log`

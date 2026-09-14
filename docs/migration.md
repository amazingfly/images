# Source migration

The original image Git history included model weights above GitHub's normal file
size limit. It is preserved intact in `.local/reorganization/original.git`.
The published repository begins with a source-only snapshot. Model files, training
data, generated media, and logs remain at their existing paths.

These source files moved into this repository. Ignored symlinks at the original
locations preserve workstation launch commands.

- `/home/derek/projects/agentic/ltxVideo/scripts/flux2_littlequeen_zerogpu` → `/mnt/storage/projects/agentic/images/scripts/flux2_littlequeen_zerogpu`
- `/home/derek/projects/agentic/ltxVideo/scripts/flux2_littlequeen_zerogpu_v2` → `/mnt/storage/projects/agentic/images/scripts/flux2_littlequeen_zerogpu_v2`
- `/home/derek/projects/agentic/ltxVideo/scripts/flux2_lora_zerogpu_eval` → `/mnt/storage/projects/agentic/images/scripts/flux2_lora_zerogpu_eval`
- `/home/derek/projects/agentic/ltxVideo/scripts/flux2_storybook_colab` → `/mnt/storage/projects/agentic/images/scripts/flux2_storybook_colab`
- `/home/derek/projects/agentic/ltxVideo/scripts/qwen35_storybook_local` → `/mnt/storage/projects/agentic/images/scripts/qwen35_storybook_local`
- `/home/derek/projects/agentic/ltxVideo/scripts/qwen35_storybook_validation` → `/mnt/storage/projects/agentic/images/scripts/qwen35_storybook_validation`
- `/home/derek/projects/agentic/ltxVideo/scripts/build_curated_storybook.py` → `/mnt/storage/projects/agentic/images/scripts/build_curated_storybook.py`
- `/home/derek/projects/agentic/ltxVideo/scripts/curated_the_witches_trick_v1.json` → `/mnt/storage/projects/agentic/images/scripts/curated_the_witches_trick_v1.json`
- `/home/derek/projects/agentic/ltxVideo/tests/test_qwen35_storybook_local.py` → `/mnt/storage/projects/agentic/images/tests/test_qwen35_storybook_local.py`
- `/home/derek/projects/agentic/ltxVideo/tests/test_qwen35_storybook_validation.py` → `/mnt/storage/projects/agentic/images/tests/test_qwen35_storybook_validation.py`
- `/home/derek/projects/agentic/sa3/sd15Files` → `/mnt/storage/projects/agentic/images/archive/sa3-sd15`

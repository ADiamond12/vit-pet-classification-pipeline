# Release Candidate Runbook

This runbook is for a local, bounded release-candidate check. It proves that the project can prepare public data, train a project-owned checkpoint, evaluate it, and produce a checksum manifest without committing data or weights to git.

It is not a public accuracy claim by itself. A prediction screenshot should only be added after the model card, evaluation artifact, and release manifest are reviewed together.

## Local Command

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_release_candidate.ps1
```

Default settings:

- dataset: Oxford-IIIT Pet through `torchvision.datasets.OxfordIIITPet`
- target: binary cats-vs-dogs labels
- subset: 50 images per class
- training: one classifier-head epoch with the encoder frozen
- outputs: ignored `data/`, `models/`, and `outputs/` folders

## Expected Artifacts

The command writes:

- `data/oxford_pet_binary_rc50/dataset_manifest.json`
- `outputs/training-rc50/training-metadata.json`
- `outputs/evaluation-rc50/eval.json`
- `outputs/evaluation-rc50/confusion_matrix.csv`
- `outputs/evaluation-rc50/evaluation-report.md`
- `outputs/release-rc50/model-release-manifest.json`
- `models/vit_catsdogs_rc50/` model files

These paths stay ignored by git. The release manifest and model files should become a GitHub Release asset only after the public model card is completed.

## Promotion Gate

Do not promote a run to a public prediction screenshot unless all checks below pass:

- dataset source and license are documented
- evaluation split and command are recorded
- model card is filled with intended use and limitations
- evaluation report and confusion matrix are reviewed
- checksum manifest covers every released model file
- no dataset, checkpoint, cache, or local output folder is committed
- metrics are strong enough to show without misleading reviewers

## Current RC50 Result

The local RC50 smoke run completed on CPU and produced the required artifact types. Its validation set is intentionally small, so it is useful as workflow evidence rather than final model evidence.

Public portfolio wording should continue to frame this repo as ML delivery proof until a stronger checkpoint and evaluation split are released.

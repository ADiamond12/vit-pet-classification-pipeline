param(
    [string]$Root = "data\raw",
    [string]$DataDir = "data\oxford_pet_binary_rc50",
    [string]$ModelDir = "models\vit_catsdogs_rc50",
    [string]$TrainingOutput = "outputs\training-rc50\training-metadata.json",
    [string]$EvaluationDir = "outputs\evaluation-rc50",
    [string]$ReleaseManifest = "outputs\release-rc50\model-release-manifest.json",
    [int]$MaxSamplesPerClass = 50,
    [int]$Epochs = 1,
    [int]$BatchSize = 4,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

Write-Host "Preparing Oxford-IIIT Pet binary subset..."
python src\training\prepare_oxford_pet_binary.py `
    --root $Root `
    --output-dir $DataDir `
    --max-samples-per-class $MaxSamplesPerClass

Write-Host "Training release-candidate checkpoint..."
python src\training\train_vit.py `
    --data-dir $DataDir `
    --output-dir $ModelDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --freeze-encoder `
    --seed $Seed `
    --metadata-output $TrainingOutput

Write-Host "Evaluating release-candidate checkpoint..."
python src\training\eval_vit.py `
    --data-dir $DataDir `
    --model-dir $ModelDir `
    --output-dir $EvaluationDir `
    --batch-size $BatchSize `
    --seed $Seed

Write-Host "Generating release checksum manifest..."
python src\training\release_manifest.py `
    --artifact-dir $ModelDir `
    --output $ReleaseManifest `
    --release-name "vit-catsdogs-rc50-local-candidate"

Write-Host "Release-candidate artifacts written under ignored data/, models/, and outputs/ folders."
Write-Host "Review metrics, model card, and release policy before publishing any checkpoint or screenshot."

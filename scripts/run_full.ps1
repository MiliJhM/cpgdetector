[CmdletBinding()]
param(
    [string]$EnvName = "cpgdetector",
    [string]$Config = "configs/default.yaml",
    [string]$RunDir = "runs/default",
    [string[]]$PredChroms = @("chr19", "chr20", "chr21", "chr22"),
    [switch]$RunDNABERT2,
    [string]$DNABERT2ModelPath = "",
    [switch]$Profile,
    [int]$ProfileBatches = 30
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $RootDir

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$LogFile = Join-Path $RunDir "full_run.log"

function Write-Log {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $Line | Tee-Object -FilePath $LogFile -Append
}

function Invoke-CondaPython {
    param([string[]]$Arguments)
    & conda run -n $EnvName python @Arguments 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: conda run -n $EnvName python $($Arguments -join ' ')"
    }
}

Write-Log "Starting full CpGDetector run"
Write-Log "Project root: $RootDir"
Write-Log "Conda env: $EnvName"
Write-Log "Config: $Config"
Write-Log "Run dir: $RunDir"
Write-Log "Prediction chromosomes: $($PredChroms -join ', ')"
Write-Log "Run DNABERT2 baseline switch: $RunDNABERT2"
Write-Log "DNABERT2 local model path override: $DNABERT2ModelPath"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH"
}

$EnvList = & conda env list
if (-not ($EnvList -match "(^|\s)$([regex]::Escape($EnvName))(\s|$)")) {
    Write-Log "Conda environment '$EnvName' not found; creating it from environment.yml"
    & conda env create -f environment.yml 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create conda environment '$EnvName'"
    }
}

$RequiredFiles = @(
    $Config,
    "data/cpgIslandExt.tsv",
    "data/dna/Homo_sapiens.GRCh38.dna.chromosome.19.fa.gz"
)
foreach ($Required in $RequiredFiles) {
    if (-not (Test-Path $Required)) {
        throw "Required file is missing: $Required"
    }
}

$ConfigDNABERT2 = & conda run -n $EnvName python -c "from cpgdetector.utils import load_yaml; import sys; cfg=load_yaml(sys.argv[1]); print('1' if cfg.get('dnabert2_baseline', {}).get('enabled', False) else '0')" $Config
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read dnabert2_baseline.enabled from config"
}
$ShouldRunDNABERT2 = [bool]$RunDNABERT2 -or (($ConfigDNABERT2 | Select-Object -Last 1).Trim() -eq "1")
Write-Log "Run DNABERT2 baseline effective: $ShouldRunDNABERT2"

Write-Log "Checking Python, PyTorch, and CUDA"
Invoke-CondaPython @(
    "-c",
    "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
)

if ($Profile) {
    Write-Log "Running throughput profile"
    Invoke-CondaPython @("-m", "cpgdetector.profile", "--config", $Config, "--batches", "$ProfileBatches")
}

Write-Log "Training model"
Invoke-CondaPython @("-m", "cpgdetector.train", "--config", $Config, "--run-dir", $RunDir)

$Checkpoint = Join-Path $RunDir "best_model.pt"
if (-not (Test-Path $Checkpoint)) {
    throw "Training did not produce checkpoint: $Checkpoint"
}

$PredArgs = @()
foreach ($Chrom in $PredChroms) {
    $PredArgs += @("--chrom", $Chrom)
}

Write-Log "Predicting CpG island BED intervals"
$PredictArgs = @(
    "-m", "cpgdetector.predict",
    "--checkpoint", $Checkpoint
) + $PredArgs + @("--out-bed", (Join-Path $RunDir "predicted_cpg_islands.bed"))
Invoke-CondaPython $PredictArgs

Write-Log "Evaluating interval-level predictions"
$EvaluateArgs = @(
    "-m", "cpgdetector.evaluate",
    "--checkpoint", $Checkpoint
) + $PredArgs + @("--output", (Join-Path $RunDir "interval_metrics.json"))
Invoke-CondaPython $EvaluateArgs

if ($ShouldRunDNABERT2) {
    $DNABERT2Dir = Join-Path $RunDir "dnabert2_baseline"
    $DNABERT2Args = @(
        "-m", "cpgdetector.dnabert2_baseline",
        "--config", $Config,
        "--output-dir", $DNABERT2Dir
    )
    if ($DNABERT2ModelPath) {
        $DNABERT2Args += @("--model-path", $DNABERT2ModelPath)
    }
    Write-Log "Training DNABERT2 window baseline"
    Invoke-CondaPython $DNABERT2Args
}

Write-Log "Regenerating baseline comparison plots"
Invoke-CondaPython @(
    "-m", "cpgdetector.compare_baselines",
    "--run-dir", $RunDir
)

Write-Log "Generating report"
Invoke-CondaPython @(
    "-m", "cpgdetector.report",
    "--run-dir", $RunDir,
    "--output", (Join-Path $RunDir "report.md")
)

Write-Log "Full run complete"
Write-Log "Key outputs:"
Write-Log "  $Checkpoint"
Write-Log "  $(Join-Path $RunDir 'metrics.csv')"
Write-Log "  $(Join-Path $RunDir 'training_curves.png')"
Write-Log "  $(Join-Path $RunDir 'baseline_comparison.png')"
Write-Log "  $(Join-Path $RunDir 'roc_pr_curves.png')"
if ($ShouldRunDNABERT2) {
    Write-Log "  $(Join-Path $RunDir 'dnabert2_baseline/dnabert2_metrics.json')"
    Write-Log "  $(Join-Path $RunDir 'dnabert2_baseline/dnabert2_roc_pr_curves.png')"
}
Write-Log "  $(Join-Path $RunDir 'predicted_cpg_islands.bed')"
Write-Log "  $(Join-Path $RunDir 'interval_metrics.json')"
Write-Log "  $(Join-Path $RunDir 'report.md')"

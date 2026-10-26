param(
    [Parameter(Mandatory=$true)][string]$TrainingJob,
    [Parameter(Mandatory=$true)][string]$EvaluationJob,
    [Parameter(Mandatory=$true)][string]$PilotProof
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonWindowless = 'E:\anaconda\envs\blind_road\pythonw.exe'
$documentPython = 'C:\Users\13699\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$prefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
foreach ($item in @($TrainingJob, $EvaluationJob, $PilotProof)) {
    if (-not ([IO.Path]::GetFullPath($item)).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'All experiment paths must stay inside attention_gate'
    }
    if ($item.Contains('"')) { throw 'Unexpected quote in experiment path' }
}
$trainScript = Join-Path $PSScriptRoot 'run_c_repeats.py'
$evalScript = Join-Path $PSScriptRoot 'evaluate_abc_sensitivity.py'
$journalScript = Join-Path $PSScriptRoot 'update_work_record.py'
$argsTrain = '"' + $trainScript + '" --mode train --job "' + $TrainingJob + '"'
$trainingProcess = Start-Process -FilePath $pythonWindowless -ArgumentList $argsTrain -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -Wait
if ($trainingProcess.ExitCode -ne 0) { throw 'C training failed; checkpoints retained and follow-up not started' }
$state = Get-Content -LiteralPath (Join-Path $TrainingJob 'job_status.json') -Encoding UTF8 -Raw | ConvertFrom-Json
if ($state.status -eq 'stopped_at_epoch_boundary') { return }
if ($state.status -ne 'complete' -or -not $state.verified) { throw 'C training did not reach verified completion' }
if (Test-Path -LiteralPath (Join-Path $TrainingJob 'STOP_AFTER_EPOCH')) { return }
$proof = Get-Content -LiteralPath $PilotProof -Encoding UTF8 -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $evalScript -Algorithm SHA256).Hash.ToLowerInvariant() -ne $proof.script_sha256) {
    throw 'Evaluation script changed after pilot; refuse automatic follow-up'
}
$pointerPath = Join-Path $PSScriptRoot 'current_training_job.json'
$pointer = [ordered]@{
    job_directory = $EvaluationJob
    C_repeats_job = $TrainingJob
    phase = 'frozen_weight_sensitivity_evaluation'
    scope = 'Nine completed A/B/C weights,500validation images each; no new training'
    ab_three_seed_study_complete = $true
    C_three_seed_study_complete = $true
    user_paused = $false
    automatic_restart = $false
    work_record = 'C:\Users\13699\Desktop\毕设\工作记录.docx'
}
$pointerTemp = $pointerPath + '.pipeline.tmp'
[IO.File]::WriteAllText($pointerTemp, ($pointer | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
Move-Item -LiteralPath $pointerTemp -Destination $pointerPath -Force
$argsEval = '"' + $evalScript + '" --mode wait-and-evaluate --source-job "' + $TrainingJob + '" --output "' + $EvaluationJob + '" --pilot-proof "' + $PilotProof + '"'
$evaluationProcess = Start-Process -FilePath $pythonWindowless -ArgumentList $argsEval -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput ($EvaluationJob + '.stdout.log') -RedirectStandardError ($EvaluationJob + '.stderr.log')
& $documentPython $journalScript --mode sensitivity-status --job $EvaluationJob
if ($evaluationProcess.ExitCode -ne 0) { throw 'Evaluation failed; partial results retained, see evaluation logs' }
# Further model experiments require completed sensitivity/diagnosis checks.
# This pipeline does not invent decisions or launch unverified candidate code.

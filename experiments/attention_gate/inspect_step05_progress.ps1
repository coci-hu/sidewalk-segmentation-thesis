param(
    [string]$JobDirectory = ''
)

$ErrorActionPreference = 'Stop'
if (-not $JobDirectory) {
    $pointerPath = Join-Path $PSScriptRoot 'current_training_job.json'
    if (-not (Test-Path -LiteralPath $pointerPath)) {
        $pointerPath = Join-Path $PSScriptRoot 'current_seed42_job.json'
    }
    $pointer = Get-Content -LiteralPath $pointerPath -Encoding UTF8 -Raw | ConvertFrom-Json
    $JobDirectory = $pointer.job_directory
}
$resolvedJob = [System.IO.Path]::GetFullPath($JobDirectory)
$workspacePrefix = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
if (-not $resolvedJob.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Job directory must be inside attention_gate'
}
$statusPath = Join-Path $resolvedJob 'job_status.json'
if (-not (Test-Path -LiteralPath $statusPath)) {
    [pscustomobject]@{ observed_status = 'initializing_or_missing_status'; job_directory = $resolvedJob } | ConvertTo-Json
    return
}
$file = Get-Item -LiteralPath $statusPath
$state = Get-Content -LiteralPath $statusPath -Encoding UTF8 -Raw | ConvertFrom-Json
$age = ((Get-Date) - $file.LastWriteTime).TotalSeconds
$observed = [string]$state.status
$alive = $false
$processStart = $null
if ($observed -in @('running', 'verifying')) {
    $taskProcess = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if ($null -ne $taskProcess -and $taskProcess.ProcessName -like 'python*') {
        $processStart = $taskProcess.StartTime
        $alive = $processStart -le $file.LastWriteTime.AddSeconds(5)
    }
    if (-not $alive) {
        $observed = 'interrupted_or_pid_reused'
    } elseif ($age -gt 120) {
        $observed = 'process_exists_status_stale'
    }
}
if ($observed -eq 'complete') {
    $proofPath = Join-Path $resolvedJob 'final_verification.json'
    $hasProof = Test-Path -LiteralPath $proofPath
    if (-not $state.verified -or -not $hasProof) {
        $observed = 'complete_not_verified'
    } elseif ((Get-Content -LiteralPath $proofPath -Encoding UTF8 -Raw | ConvertFrom-Json).status -ne 'pass') {
        $observed = 'complete_not_verified'
    }
}
$savedEpoch = $state.saved_completed_epoch
if ($state.variant) {
    $folder = if ($state.variant_folder) { $state.variant_folder } else { Join-Path $resolvedJob ($state.variant + '_seed42_epochs002_100') }
    $savedPath = Join-Path $folder 'status.json'
    if (Test-Path -LiteralPath $savedPath) {
        $savedEpoch = (Get-Content -LiteralPath $savedPath -Encoding UTF8 -Raw | ConvertFrom-Json).completed_epoch
    }
}
[pscustomobject]@{
    observed_status = $observed
    recorded_status = $state.status
    checked_at = (Get-Date).ToString('o')
    status_updated_at = $file.LastWriteTime.ToString('o')
    status_age_seconds = [math]::Round($age, 1)
    parent_process_alive_and_not_newer_than_status = $alive
    parent_process_started_at = $processStart
    variant = $state.variant
    seed = $state.seed
    saved_completed_epoch = $savedEpoch
    training = $state.training
    phase = if ($state.phase) { $state.phase } elseif ($null -ne $pointer) { $pointer.phase } else { $null }
    evaluation_images_completed = $state.images_completed
    evaluation_images_total = $state.images_total
    evaluation_models_completed = $state.completed_models
    verified = $state.verified
    job_directory = $resolvedJob
    completed_runs_in_current_job = $state.completed_runs
    total_runs_in_current_job = $state.total_runs
    three_seed_study_complete = [bool]$state.three_seed_study_complete
    ab_three_seed_study_complete = if ($null -ne $state.ab_three_seed_study_complete) { [bool]$state.ab_three_seed_study_complete } elseif ($null -ne $pointer) { [bool]$pointer.ab_three_seed_study_complete } else { $false }
    C_three_seed_study_complete = if ($null -ne $state.C_three_seed_study_complete) { [bool]$state.C_three_seed_study_complete } elseif ($null -ne $pointer) { [bool]$pointer.C_three_seed_study_complete } else { $false }
} | ConvertTo-Json -Depth 10

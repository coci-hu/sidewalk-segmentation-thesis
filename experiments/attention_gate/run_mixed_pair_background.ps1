param([Parameter(Mandatory=$true)][string]$TrainingJob)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonWindowless = 'E:\anaconda\envs\blind_road\pythonw.exe'
$resolvedJob = [IO.Path]::GetFullPath($TrainingJob)
$allowedPrefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
if (-not $resolvedJob.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase) -or $resolvedJob.Contains('"')) {
    throw 'Training job must stay inside the experiment directory'
}
$statusPath = Join-Path $resolvedJob 'job_status.json'
$state = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($state.status -ne 'prepared') { throw 'Only a newly prepared job may be launched; no automatic retry' }
$outLog = Join-Path $resolvedJob 'supervisor.stdout.log'
$errLog = Join-Path $resolvedJob 'supervisor.stderr.log'
if ((Test-Path -LiteralPath $outLog) -or (Test-Path -LiteralPath $errLog)) { throw 'Launch logs already exist' }
$runner = Join-Path $PSScriptRoot 'run_mixed_pair.py'
$trainingArguments = '"' + $runner + '" --mode train --job "' + $resolvedJob + '"'
$trainingProcess = Start-Process -FilePath $pythonWindowless -ArgumentList $trainingArguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $outLog -RedirectStandardError $errLog
if ($trainingProcess.ExitCode -ne 0) { throw 'D/E supervisor failed; outputs retained and no retry started' }
$state = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($state.status -eq 'stopped_at_epoch_boundary') { return }
if ($state.status -ne 'complete' -or -not $state.verified) { throw 'Training did not reach verified completion' }

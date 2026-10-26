param(
    [string]$Script,
    [string]$Python = 'E:\anaconda\envs\blind_road\python.exe',
    [string[]]$ScriptArgs = @(),
    [switch]$List,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$taskProject = [IO.Path]::GetFullPath($PSScriptRoot)
$taskCode = Join-Path $taskProject 'code'
$taskSam = Join-Path $taskProject 'code_sam'
if ($List) {
    Get-ChildItem -LiteralPath $taskCode,$taskSam -Filter '*.py' -File |
        Where-Object { $_.Name -match '^(train_|eval_|benchmark_|deploy_|plot_|visualize_|video_|sam_|labelme_|split_|compare_|copy_|test_)' } |
        ForEach-Object { $_.FullName.Substring($taskProject.Length + 1) }
    Write-Output 'code\pipeline\train_main.py'
    exit 0
}
if (-not $Script) { throw 'Use -List, or specify -Script and optionally -DryRun.' }
$taskCandidates = @()
foreach ($taskBase in @($taskProject, $taskCode, $taskSam)) {
    $taskPath = [IO.Path]::GetFullPath((Join-Path $taskBase $Script))
    $taskAllowed = $taskPath.StartsWith($taskCode + '\', [StringComparison]::OrdinalIgnoreCase) -or
                   $taskPath.StartsWith($taskSam + '\', [StringComparison]::OrdinalIgnoreCase)
    if ($taskAllowed -and [IO.Path]::GetExtension($taskPath) -eq '.py' -and (Test-Path -LiteralPath $taskPath -PathType Leaf)) {
        $taskCandidates += $taskPath
    }
}
$taskCandidates = @($taskCandidates | Select-Object -Unique)
if ($taskCandidates.Count -ne 1) { throw "Script is missing or ambiguous: $Script" }
$taskScript = $taskCandidates[0]
$taskWorking = if ($taskScript.StartsWith($taskSam + '\', [StringComparison]::OrdinalIgnoreCase)) { $taskProject } else { $taskCode }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable not found. Supply -Python with your environment path: $Python" }
if ($DryRun) {
    Write-Output ("Python: {0}" -f $Python)
    Write-Output ("Script: {0}" -f $taskScript)
    Write-Output ("Working directory: {0}" -f $taskWorking)
    Write-Output ("Arguments: {0}" -f ($ScriptArgs -join ' '))
    Write-Output 'Dry run only; script not executed.'
    exit 0
}
Push-Location -LiteralPath $taskWorking
try {
    & $Python -B $taskScript @ScriptArgs
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode

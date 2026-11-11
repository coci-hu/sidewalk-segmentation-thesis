param(
    [string]$Python = 'python',
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$PackagingTools = ''
)
$ErrorActionPreference='Stop'
$destination=[System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $destination) { throw 'Choose a new output directory; existing builds are preserved.' }
New-Item -ItemType Directory -Path $destination | Out-Null
$priorPythonPath=$env:PYTHONPATH
try {
    if ($PackagingTools) { $env:PYTHONPATH=$PackagingTools }
    & $Python -m PyInstaller --distpath (Join-Path $destination 'dist') --workpath (Join-Path $destination 'build') (Join-Path $PSScriptRoot 'gpu_windows.spec')
    if ($LASTEXITCODE -ne 0) { throw 'GPU executable build failed. Keep logs and inspect before distribution.' }
} finally {
    $env:PYTHONPATH=$priorPythonPath
}

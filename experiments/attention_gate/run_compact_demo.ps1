param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [ValidateSet('A','L0')][string]$Model='L0',
    [string]$Python='E:\anaconda\envs\blind_road\python.exe',
    [string]$ModelsPath=(Join-Path $PSScriptRoot 'step_13_2026-11-07\A_L0_deployment_001\export\models.json'),
    [switch]$Headless
)
$ErrorActionPreference='Stop'
$inputFile=(Resolve-Path -LiteralPath $InputPath).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Set -Python to the verified environment interpreter.' }
$demoFolder=Join-Path $PSScriptRoot ('demo_outputs\compact_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $demoFolder | Out-Null
$extension=[IO.Path]::GetExtension($inputFile).ToLowerInvariant()
$isImage=$extension -in @('.jpg','.jpeg','.png','.bmp','.webp')
$outputFile=Join-Path $demoFolder $(if ($isImage) {'overlay.png'} else {'overlay.mp4'})
$modelIndex=@{A=0;L0=1}[$Model]
$demoArguments=@((Join-Path $PSScriptRoot 'sidewalk_demo_compact.py'),'--models',$ModelsPath,'--input',$inputFile,
    '--output',$outputFile,'--backend','cuda','--model-index',[string]$modelIndex)
if ($Headless) { $demoArguments+='--headless' }
& $Python @demoArguments
if ($LASTEXITCODE -ne 0) { throw "Demo failed; inspect output folder $demoFolder" }
Write-Output "Output: $outputFile"

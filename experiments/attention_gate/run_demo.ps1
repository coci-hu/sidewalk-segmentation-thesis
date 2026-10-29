param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [ValidateSet('A','B','C')][string]$Model='B',
    [string]$Python='E:\anaconda\envs\blind_road\python.exe',
    [string]$ModelsPath=(Join-Path $PSScriptRoot 'step_10_2026-10-28\preserved_bn_reexport_cpu_001\models.json'),
    [switch]$Headless
)
$ErrorActionPreference='Stop'
$inputFile=(Resolve-Path -LiteralPath $InputPath).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Set -Python to the verified environment interpreter.' }
$demoFolder=Join-Path $PSScriptRoot ('demo_outputs\'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $demoFolder | Out-Null
$extension=[IO.Path]::GetExtension($inputFile).ToLowerInvariant()
$isImage=$extension -in @('.jpg','.jpeg','.png','.bmp','.webp')
$outputFile=Join-Path $demoFolder $(if ($isImage) {'overlay.png'} else {'overlay.mp4'})
$modelIndex=@{A=0;B=1;C=2}[$Model]
$demoArguments=@((Join-Path $PSScriptRoot 'sidewalk_demo.py'),'--models',$ModelsPath,'--input',$inputFile,
    '--output',$outputFile,'--backend','cuda','--model-index',[string]$modelIndex)
if ($Headless) { $demoArguments+='--headless' }
& $Python @demoArguments
if ($LASTEXITCODE -ne 0) { throw "Demo failed; inspect output folder $demoFolder" }
Write-Output "Output: $outputFile"

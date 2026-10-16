param([string]$Script='build-landing.js', [switch]$IncludeData)
$ErrorActionPreference='Stop'
$taskDir=$PSScriptRoot
$taskPen=Join-Path $taskDir 'landing.pen'
$taskCode=Get-Content -LiteralPath (Join-Path $taskDir $Script) -Raw
if($IncludeData) {
    $taskData=Get-Content -LiteralPath (Join-Path $taskDir 'visual-data.json') -Raw
    $taskCode='const D='+$taskData+';'+[Environment]::NewLine+$taskCode
}
$taskCommand='execute({input:'+($taskCode | ConvertTo-Json -Compress)+'})'
$taskCli='C:/dev/mgts/design/pen-cli-validation/node_modules/@pen.dev/cli/dist/index.mjs'
$taskLog=Join-Path $taskDir 'last-cli.log'
@($taskCommand,'save()','exit()') | node $taskCli interactive -i $taskPen -o $taskPen *> $taskLog
Get-Content -LiteralPath $taskLog | Where-Object { $_.Length -lt 1300 -and $_ -notmatch 'image.*iVBOR|mimeType' } | Select-Object -Last 150
if(Select-String -LiteralPath $taskLog -SimpleMatch 'Failure during operation execution' -Quiet) { exit 2 }
exit $LASTEXITCODE

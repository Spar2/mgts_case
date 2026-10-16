param([string]$Records = "$PSScriptRoot/../.runtime/records", [string]$Queue = "$PSScriptRoot/../.runtime/queue", [string]$ResearchPack = '')
$taskRoot = (Resolve-Path "$PSScriptRoot/..").Path
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
if (!(Test-Path $taskPython)) { throw 'Create .venv and install requirements first; see docs/web/README.md' }
if (!(Test-Path "$Queue/queue.json")) { throw 'Queue directory must contain queue.json and manifest.json' }
$env:DTS_RECORDS = (Resolve-Path $Records).Path
$env:DTS_QUEUE = (Resolve-Path $Queue).Path
if ($ResearchPack) { $env:DTS_RESEARCH_PACK = (Resolve-Path $ResearchPack).Path } else { $env:DTS_RESEARCH_PACK = '' }
$env:DTS_RUNTIME = Join-Path $taskRoot '.runtime'
$env:PYTHONPATH = Join-Path $taskRoot 'services/web-api'
$taskLogs = Join-Path $taskRoot '.runtime/logs'
New-Item -ItemType Directory -Force $taskLogs | Out-Null
$taskApi = Start-Process -FilePath $taskPython -ArgumentList '-m','uvicorn','workbench.api:app','--host','127.0.0.1','--port','8091' -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$taskLogs/api.log" -RedirectStandardError "$taskLogs/api-error.log"
try {
  $taskReady = $false
  for ($taskAttempt=0; $taskAttempt -lt 60; $taskAttempt++) {
    if ($taskApi.HasExited) { throw "API exited; inspect $taskLogs/api-error.log" }
    try { $null = Invoke-RestMethod http://127.0.0.1:8091/api/v1/health -TimeoutSec 1; $taskReady=$true; break } catch { Start-Sleep -Milliseconds 500 }
  }
  if (!$taskReady) { throw "API startup timed out; inspect $taskLogs/api-error.log" }
  Set-Location "$taskRoot/apps/web"
  npm.cmd run dev
} finally { if (!$taskApi.HasExited) { Stop-Process -Id $taskApi.Id } }

# Competition leaderboard — hourly capture + CDN publish (Windows Task Scheduler entry point).
# Read-only over participant wallets; isolated from the live trading agent. Logs to data/competition_out/hourly.log.
$ErrorActionPreference = 'Continue'
$proj = 'E:\projects\agentic-crypto-trader'
Set-Location $proj
$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'

$logDir = Join-Path $proj 'data\competition_out'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$log = Join-Path $logDir 'hourly.log'
$start = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
"=== $start capture start ===" | Out-File -Append -FilePath $log -Encoding utf8

& "$proj\.venv\Scripts\python.exe" -m trader.competition `
    --out data\competition_out `
    --publish s3://alexlouis-apentic-data *>> $log

$code = $LASTEXITCODE
$end = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
"=== $end capture done (exit $code) ===" | Out-File -Append -FilePath $log -Encoding utf8
exit $code

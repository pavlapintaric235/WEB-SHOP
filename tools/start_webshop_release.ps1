$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$releaseDataPath = Join-Path $workspacePath 'data\webshop_release_demo'
if (-not (Test-Path -LiteralPath (Join-Path $releaseDataPath 'inventorylens.sqlite3'))) {
    throw 'Prepare the isolated release demo first: .\.venv\Scripts\python.exe tools\prepare_webshop_release.py'
}
$env:WEBSHOP_DATA_DIR = $releaseDataPath
# Match the CPU settings used by the successful connected scan smoke test.
$cpuThreads = if ($env:WEBSHOP_CPU_THREADS) { $env:WEBSHOP_CPU_THREADS } else { '2' }
foreach ($threadSetting in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')) {
    [Environment]::SetEnvironmentVariable($threadSetting, $cpuThreads, 'Process')
}
$connectionFile = Join-Path $workspacePath 'data\webshop_demo\worker_connection.json'
if (Test-Path -LiteralPath $connectionFile) {
    $connection = Get-Content -LiteralPath $connectionFile -Raw | ConvertFrom-Json
    $env:WEBSHOP_DINO_URL = $connection.url
    $env:WEBSHOP_DINO_TOKEN = $connection.token
}
& (Join-Path $workspacePath '.venv\Scripts\python.exe') -m streamlit run (Join-Path $workspacePath 'app.py') --server.port 8508 --server.address 127.0.0.1

$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$env:WEBSHOP_DATA_DIR = Join-Path $workspacePath 'data\webshop_ux_demo'
if (-not (Test-Path -LiteralPath (Join-Path $env:WEBSHOP_DATA_DIR 'inventorylens.sqlite3'))) {
    throw 'Run .\.venv\Scripts\python.exe tools\prepare_webshop_ux_demo.py first.'
}
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
& (Join-Path $workspacePath '.venv\Scripts\python.exe') -m streamlit run (Join-Path $workspacePath 'app_8509.py') --server.port 8509 --server.address 127.0.0.1 --server.fileWatcherType none --theme.base light --theme.primaryColor '#126552'

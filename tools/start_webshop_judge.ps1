param([string]$PublicHostname = '')
$ErrorActionPreference = 'Stop'
if ($PublicHostname) {
    if ($PublicHostname -notmatch '^[a-zA-Z0-9.-]+$') { throw 'Supply a hostname without protocol, port or path.' }
    $env:STREAMLIT_BROWSER_SERVER_ADDRESS = $PublicHostname
    $env:STREAMLIT_BROWSER_SERVER_PORT = '443'
}
$workspacePath = Split-Path -Parent $PSScriptRoot
$accessFile = Join-Path $workspacePath 'data\judge_release\access.json'
if (-not (Test-Path -LiteralPath $accessFile)) { throw 'Run .\.venv\Scripts\python.exe tools\prepare_judge_demo.py first.' }
$access = Get-Content -LiteralPath $accessFile -Raw | ConvertFrom-Json
$env:WEBSHOP_JUDGE_ACCESS_CODE = $access.access_code
$env:WEBSHOP_JUDGE_MODE = '1'
$env:WEBSHOP_DATA_DIR = Join-Path $workspacePath 'data\webshop_judge_demo'
foreach ($setting in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) { [Environment]::SetEnvironmentVariable($setting,'2','Process') }
$connectionFile = Join-Path $workspacePath 'data\webshop_demo\worker_connection.json'
if (Test-Path -LiteralPath $connectionFile) {
    $connection = Get-Content -LiteralPath $connectionFile -Raw | ConvertFrom-Json
    $env:WEBSHOP_DINO_URL = $connection.url
    $env:WEBSHOP_DINO_TOKEN = $connection.token
}
& (Join-Path $workspacePath '.venv\Scripts\python.exe') -m streamlit run (Join-Path $workspacePath 'app_judge.py') --server.port 8511 --server.address 127.0.0.1 --server.fileWatcherType none --server.enableCORS true --server.enableXsrfProtection true --server.enableStaticServing false --server.maxUploadSize 20 --client.showErrorDetails false --client.toolbarMode minimal --browser.gatherUsageStats false --theme.base light --theme.primaryColor '#126552'

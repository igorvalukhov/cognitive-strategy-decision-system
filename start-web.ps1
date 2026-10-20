param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$webPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $webPython)) {
    throw 'Create a Python 3.10+ virtual environment first: py -m venv .venv. Then install requirements-module1.txt.'
}
Write-Host "Open http://127.0.0.1:$Port in your browser. Press Ctrl+C to stop."
& $webPython -m uvicorn src.web.app:app --host 127.0.0.1 --port $Port


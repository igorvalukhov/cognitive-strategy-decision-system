param(
    [Parameter(Mandatory=$true)][string]$ServerPath,
    [Parameter(Mandatory=$true)][string]$ModelPath
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ServerPath -PathType Leaf)) { throw 'llama-server executable not found' }
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw 'GigaChat GGUF weights not found' }
& $ServerPath -m $ModelPath --alias gigachat-local --host 127.0.0.1 --port 8081 -c 4096

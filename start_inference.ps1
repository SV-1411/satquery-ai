param([int]$Port = 8000)
Set-Location $PSScriptRoot
$torchLib = Join-Path $PSScriptRoot '.venv\Lib\site-packages\torch\lib'
if(Test-Path -LiteralPath $torchLib) { $env:PATH = "$torchLib;$env:PATH" }
& .\.venv\Scripts\python.exe -m uvicorn inference.api:app --host 127.0.0.1 --port $Port

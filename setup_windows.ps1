param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
Set-Location -LiteralPath $taskRoot

function Get-TaskSha256([string]$LiteralPath) {
    $taskHasher = [Security.Cryptography.SHA256]::Create()
    $taskStream = [IO.File]::OpenRead($LiteralPath)
    try { return [BitConverter]::ToString($taskHasher.ComputeHash($taskStream)).Replace('-', '').ToLowerInvariant() }
    finally { $taskStream.Dispose(); $taskHasher.Dispose() }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows 64 bits est requis.' }
    $taskArch = $env:PROCESSOR_ARCHITEW6432
    if (-not $taskArch) { $taskArch = $env:PROCESSOR_ARCHITECTURE }
    if ($taskArch -ne 'AMD64') { throw 'Cette installation autonome cible Windows x64 (Intel/AMD).' }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $taskPythonRoot = Join-Path $taskRoot '.runtimes\python'
    $taskPython = Join-Path $taskPythonRoot 'python.exe'

    if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
        $taskTar = Get-Command tar.exe -ErrorAction Stop
        $taskDownloads = Join-Path $taskRoot '.runtimes\downloads'
        New-Item -ItemType Directory -Path $taskDownloads -Force | Out-Null
        $taskArchive = Join-Path $taskDownloads 'cpython-3.12.15-windows-x64.tar.gz'
        $taskExpected = 'bbc902d4d187ac4914915bf538196a064b49109a33a104965b5164ced5b80a1e'
        $taskUrl = 'https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15%2B20261013-x86_64-pc-windows-msvc-install_only.tar.gz'
        $taskValid = (Test-Path -LiteralPath $taskArchive -PathType Leaf) -and ((Get-TaskSha256 $taskArchive) -eq $taskExpected)
        if (-not $taskValid) {
            Write-Host '[1/4] Telechargement du Python portable (environ 46 Mo)...'
            $taskOldProgress = $ProgressPreference
            $ProgressPreference = 'SilentlyContinue'
            try { Invoke-WebRequest -Uri $taskUrl -OutFile ($taskArchive + '.part') -UseBasicParsing }
            finally { $ProgressPreference = $taskOldProgress }
            if ((Get-TaskSha256 ($taskArchive + '.part')) -ne $taskExpected) { throw 'SHA-256 du Python portable incorrect.' }
            Move-Item -LiteralPath ($taskArchive + '.part') -Destination $taskArchive -Force
        }
        $taskStage = Join-Path $taskRoot ('.runtimes\python-stage-' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $taskStage | Out-Null
        & $taskTar.Source -xzf $taskArchive -C $taskStage
        if ($LASTEXITCODE -ne 0) { throw 'Extraction du Python portable impossible.' }
        $taskExtracted = Join-Path $taskStage 'python'
        if (-not (Test-Path -LiteralPath (Join-Path $taskExtracted 'python.exe'))) { throw 'Python absent de l archive.' }
        if (Test-Path -LiteralPath $taskPythonRoot) { throw 'Installation Python partielle detectee dans .runtimes\python ; ne pas ecraser ce dossier.' }
        $taskResolvedSource = [IO.Path]::GetFullPath($taskExtracted)
        $taskResolvedTarget = [IO.Path]::GetFullPath($taskPythonRoot)
        $taskRuntimePrefix = [IO.Path]::GetFullPath((Join-Path $taskRoot '.runtimes')) + [IO.Path]::DirectorySeparatorChar
        if (-not $taskResolvedSource.StartsWith($taskRuntimePrefix, [StringComparison]::OrdinalIgnoreCase) -or -not $taskResolvedTarget.StartsWith($taskRuntimePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Chemin Python hors du dossier attendu.' }
        Move-Item -LiteralPath $taskResolvedSource -Destination $taskResolvedTarget
        Remove-Item -LiteralPath $taskStage
    }

    Write-Host '[2/4] Installation des dependances Python locales...'
    & $taskPython -I -m pip install --disable-pip-version-check --no-warn-script-location -r (Join-Path $taskRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Installation des dependances impossible.' }
    Write-Host '[3/4] Installation de Qwen 2B Heretic et de llama.cpp (environ 1,35 Go)...'
    $taskInstallArgs = @((Join-Path $taskRoot 'install_local.py'))
    if ($CpuOnly) { $taskInstallArgs += @('--backend', 'cpu') }
    & $taskPython @taskInstallArgs
    if ($LASTEXITCODE -ne 0) { throw 'Installation ou verification de Qwen impossible.' }
    Write-Host '[4/4] Installation autonome terminee. Double-clique sur lancer.bat.'
    exit 0
} catch {
    Write-Host ('ERREUR : ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Corrige le probleme puis relance install.bat. Les fichiers valides seront reutilises.'
    exit 1
}

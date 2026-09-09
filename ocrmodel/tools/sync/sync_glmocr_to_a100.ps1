[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9._@-]+$')]
    [string]$RemoteHost,

    [Parameter(Mandatory)]
    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot,

    [string]$RepositoryRoot = '',

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryPath = $RepositoryRoot
if ([string]::IsNullOrWhiteSpace($repositoryPath)) {
    $repositoryPath = Join-Path $PSScriptRoot '..\..'
}
$repository = (Resolve-Path -LiteralPath $repositoryPath).Path
$gitRoot = (Resolve-Path (Join-Path $repository '..')).Path
if ($RemoteRoot -eq '/' -or $RemoteRoot.EndsWith('/..')) {
    throw "Unsafe remote root: $RemoteRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $repository 'README.md') -PathType Leaf)) {
    throw "Repository root does not contain ocrmodel/README.md: $repository"
}

$requiredCommands = @('git.exe')
if (-not $DryRun) {
    $requiredCommands += 'ssh.exe', 'scp.exe'
}
foreach ($commandName in $requiredCommands) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command is not available: $commandName"
    }
}

$relativeFiles = @(
    & git.exe -C $gitRoot ls-files --cached --others --exclude-standard -- `
        ocrmodel/src ocrmodel/tools ocrmodel/config
)
if ($LASTEXITCODE -ne 0) {
    throw 'git ls-files failed.'
}
$relativeFiles = @(
    $relativeFiles |
        Where-Object { $_ -and $_ -ne 'ocrmodel/config/paths.env' } |
        ForEach-Object { $_.Substring('ocrmodel/'.Length) } |
        Sort-Object -Unique
)
if ($relativeFiles.Count -eq 0) {
    throw 'No synchronized source files were found.'
}

$temporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
)
$temporaryRoot = Join-Path $temporaryBase ("ocrmodel-glmocr-sync-" + [guid]::NewGuid().ToString('N'))
$repositoryPrefix = $repository.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar

try {
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    foreach ($relativeFile in $relativeFiles) {
        $nativeRelative = $relativeFile.Replace('/', [IO.Path]::DirectorySeparatorChar)
        $source = [IO.Path]::GetFullPath((Join-Path $repository $nativeRelative))
        if (-not $source.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to copy a path outside the repository: $relativeFile"
        }
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Git-visible source file is missing: $relativeFile"
        }
        $destination = Join-Path $temporaryRoot $nativeRelative
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }

    if ($DryRun) {
        Write-Output "SYNC_DRY_RUN_OK files=$($relativeFiles.Count)"
        return
    }

    & ssh.exe $RemoteHost "mkdir -p -- '$RemoteRoot'"
    if ($LASTEXITCODE -ne 0) {
        throw 'Remote directory creation failed.'
    }
    $uploadRoots = @('src', 'tools', 'config') |
        ForEach-Object { Join-Path $temporaryRoot $_ } |
        Where-Object { Test-Path -LiteralPath $_ }
    & scp.exe -r @uploadRoots "${RemoteHost}:${RemoteRoot}/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Source synchronization failed.'
    }
    Write-Output "SYNC_OK files=$($relativeFiles.Count) remote=${RemoteHost}:${RemoteRoot}"
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        $resolvedTemporary = (Resolve-Path -LiteralPath $temporaryRoot).Path
        $temporaryPrefix = $temporaryBase + [IO.Path]::DirectorySeparatorChar
        if (-not $resolvedTemporary.StartsWith($temporaryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove unexpected temporary path: $resolvedTemporary"
        }
        Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
    }
}

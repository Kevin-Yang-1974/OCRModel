[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9._@-]+$')]
    [string]$RemoteHost,

    [Parameter(Mandatory)]
    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot,

    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
if ($RemoteRoot -eq '/' -or $RemoteRoot.EndsWith('/..')) {
    throw "Unsafe remote root: $RemoteRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $repository 'README.md') -PathType Leaf)) {
    throw "Repository root does not contain README.md: $repository"
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
    & git.exe -C $repository ls-files --cached --others --exclude-standard -- `
        src tools config references
)
if ($LASTEXITCODE -ne 0) {
    throw 'git ls-files failed.'
}
$relativeFiles = @(
    $relativeFiles |
        Where-Object { $_ -and $_ -ne 'config/paths.env' } |
        Sort-Object -Unique
)
if ($relativeFiles.Count -eq 0) {
    throw 'No synchronized source files were found.'
}

$temporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
)
$temporaryRoot = Join-Path $temporaryBase ("ocrmodel-sync-" + [guid]::NewGuid().ToString('N'))
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
        $destinationDirectory = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
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

    $uploadRoots = @(
        'src', 'tools', 'config', 'references' |
            ForEach-Object { Join-Path $temporaryRoot $_ } |
            Where-Object { Test-Path -LiteralPath $_ }
    )
    $destinationArgument = "${RemoteHost}:${RemoteRoot}/"
    & scp.exe -r @uploadRoots $destinationArgument
    if ($LASTEXITCODE -ne 0) {
        throw 'Source synchronization failed.'
    }

    Write-Output "SYNC_OK files=$($relativeFiles.Count) remote=${RemoteHost}:${RemoteRoot}"
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        $resolvedTemporary = (Resolve-Path -LiteralPath $temporaryRoot).Path
        $temporaryPrefix = $temporaryBase + [IO.Path]::DirectorySeparatorChar
        if (-not $resolvedTemporary.StartsWith(
            $temporaryPrefix,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refusing to remove unexpected temporary path: $resolvedTemporary"
        }
        Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
    }
}

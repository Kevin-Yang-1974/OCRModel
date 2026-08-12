[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9._@-]+$')]
    [string]$RemoteHost,

    [Parameter(Mandatory)]
    [string]$LocalDatasetRoot,

    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteLayoutDataRoot = '/data3/yky/yangky_ocr_models/training_data/got_layout_pages',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$DatasetId,

    [switch]$AllowExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$localRoot = (Resolve-Path -LiteralPath $LocalDatasetRoot).Path
if (-not (Test-Path -LiteralPath $localRoot -PathType Container)) {
    throw "Local dataset root is not a directory: $localRoot"
}
if (-not $DatasetId) {
    $DatasetId = Split-Path -Leaf $localRoot
}
if (-not $DatasetId -or $DatasetId -notmatch '^[A-Za-z0-9._-]+$' -or $DatasetId -eq '.' -or $DatasetId -eq '..') {
    throw "Unsafe dataset id: $DatasetId"
}
$localDatasetId = Split-Path -Leaf $localRoot
if ($DatasetId -ne $localDatasetId) {
    throw "DatasetId must match the local dataset directory name for scp -r upload: local=$localDatasetId requested=$DatasetId"
}

$requiredFiles = @(
    'train/manifest.jsonl',
    'validation/manifest.jsonl',
    'test/manifest.jsonl',
    'split_audit.json'
)
foreach ($relative in $requiredFiles) {
    $nativeRelative = $relative.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $path = Join-Path $localRoot $nativeRelative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Formal layout dataset is missing required file: $relative"
    }
}

foreach ($commandName in @('ssh.exe', 'scp.exe')) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command is not available: $commandName"
    }
}

if ($RemoteLayoutDataRoot -eq '/' -or $RemoteLayoutDataRoot.EndsWith('/..')) {
    throw "Unsafe remote layout data root: $RemoteLayoutDataRoot"
}
$remoteDatasetRoot = "$RemoteLayoutDataRoot/$DatasetId"
$existenceCheck = if ($AllowExisting) {
    "mkdir -p -- '$RemoteLayoutDataRoot' '$remoteDatasetRoot'"
} else {
    "mkdir -p -- '$RemoteLayoutDataRoot' && test ! -e '$remoteDatasetRoot'"
}

& ssh.exe $RemoteHost $existenceCheck
if ($LASTEXITCODE -ne 0) {
    throw "Remote dataset already exists or parent creation failed: ${RemoteHost}:$remoteDatasetRoot"
}

& scp.exe -r $localRoot "${RemoteHost}:$RemoteLayoutDataRoot/"
if ($LASTEXITCODE -ne 0) {
    throw 'Layout dataset upload failed.'
}

Write-Output "LAYOUT_DATASET_UPLOAD_OK dataset_id=$DatasetId remote=${RemoteHost}:$remoteDatasetRoot"

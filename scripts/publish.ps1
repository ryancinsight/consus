<#
.SYNOPSIS
Publishes the consus workspace crates to crates.io in topological order.

.DESCRIPTION
Because consus is a multi-crate workspace, `cargo publish --workspace` will fail if leaf
crates rely on internal dependencies that haven't propagated to the crates.io index yet.
This script automates the topological release sequence.

.PARAMETER DryRun
If set, runs `cargo publish --dry-run` instead of actually publishing.
NOTE: `--dry-run` will fail on dependent crates (like consus-compression) because their dependencies (like consus-core) are not actually uploaded to crates.io during a dry run.

.EXAMPLE
.\scripts\publish.ps1 -DryRun
#>
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Define the topological order of crates
$Order = @(
    "consus-core",
    "consus-compression",
    "consus-io",
    "consus-arrow",
    "consus-hdf5",
    "consus-mat",
    "consus-zarr",
    "consus-fits",
    "consus-parquet",
    "consus-netcdf",
    "consus-nwb",
    "consus"
)

Write-Host "Starting consus topological release..." -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "Running in DRY RUN mode" -ForegroundColor Yellow
}

foreach ($crate in $Order) {
    Write-Host "`n========================================" -ForegroundColor Magenta
    Write-Host "Publishing: $crate" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Magenta
    
    $cmd = "cargo publish -p $crate"
    if ($DryRun) {
        $cmd += " --dry-run"
    }

    Write-Host "> $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to publish $crate. Aborting topological release."
        exit $LASTEXITCODE
    }

    if (-not $DryRun -and $crate -ne "consus") {
        Write-Host "Waiting 15 seconds for crates.io index propagation..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
    }
}

Write-Host "`nRelease complete!" -ForegroundColor Cyan

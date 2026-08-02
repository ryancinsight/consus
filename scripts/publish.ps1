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
Set-StrictMode -Version Latest

function Get-PublishOrder {
    $metadataJson = & cargo metadata --locked --no-deps --format-version 1
    if ($LASTEXITCODE -ne 0) {
        throw "cargo metadata failed with exit code $LASTEXITCODE"
    }

    $packages = @($metadataJson | ConvertFrom-Json | Select-Object -ExpandProperty packages |
        Where-Object { $null -eq $_.publish })
    $byName = @{}
    foreach ($package in $packages) {
        $byName[$package.name] = $package
    }

    $dependencies = @{}
    foreach ($package in $packages) {
        $dependencies[$package.name] = @($package.dependencies |
            Where-Object {
                $pathProperty = $_.PSObject.Properties["path"]
                $_.kind -ne "dev" -and
                $null -ne $pathProperty -and
                $null -ne $pathProperty.Value -and
                $byName.ContainsKey($_.name)
            } |
            ForEach-Object { $_.name } |
            Sort-Object -Unique)
    }

    $order = [System.Collections.Generic.List[string]]::new()
    while ($order.Count -lt $packages.Count) {
        $ready = @($packages.name |
            Where-Object {
                $_ -notin $order -and
                @($dependencies[$_] | Where-Object { $_ -notin $order }).Count -eq 0
            } |
            Sort-Object)
        if ($ready.Count -eq 0) {
            throw "Publishable workspace dependencies contain a cycle"
        }
        foreach ($name in $ready) {
            $order.Add($name)
        }
    }

    return $order
}

function Wait-CrateVersion {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Version
    )

    $uri = "https://crates.io/api/v1/crates/$Name/$Version"
    foreach ($attempt in 1..24) {
        try {
            Invoke-RestMethod -Uri $uri -TimeoutSec 10 | Out-Null
            return
        }
        catch {
            if ($attempt -eq 24) {
                throw "crates.io did not expose $Name $Version within 120 seconds"
            }
            Start-Sleep -Seconds 5
        }
    }
}

$Metadata = & cargo metadata --locked --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "cargo metadata failed with exit code $LASTEXITCODE"
}
$Versions = @{}
foreach ($package in $Metadata.packages) {
    $Versions[$package.name] = $package.version
}
$Order = @(Get-PublishOrder)

Write-Host "Starting consus topological release..." -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "Running in DRY RUN mode" -ForegroundColor Yellow
}

foreach ($crate in $Order) {
    Write-Host "`n========================================" -ForegroundColor Magenta
    Write-Host "Publishing: $crate" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Magenta
    
    $arguments = @("publish", "--locked", "--package", $crate)
    if ($DryRun) {
        $arguments += "--dry-run"
    }

    Write-Host "> cargo $($arguments -join ' ')" -ForegroundColor DarkGray
    & cargo @arguments

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to publish $crate. Aborting topological release."
        exit $LASTEXITCODE
    }

    if (-not $DryRun) {
        Write-Host "Waiting for crates.io to expose $crate $($Versions[$crate])..." -ForegroundColor Yellow
        Wait-CrateVersion -Name $crate -Version $Versions[$crate]
    }
}

Write-Host "`nRelease complete!" -ForegroundColor Cyan

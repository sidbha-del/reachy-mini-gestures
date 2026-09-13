<#
.SYNOPSIS
  Check / start / stop / restart the Reachy Mini daemon + gesture web app.

.USAGE
  .\manage.ps1              (interactive menu)
  .\manage.ps1 status
  .\manage.ps1 start
  .\manage.ps1 stop
  .\manage.ps1 restart
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("status", "start", "stop", "restart", "menu")]
    [string]$Action = "menu"
)

$ErrorActionPreference = "Stop"

$ProjectDir  = $PSScriptRoot
$DaemonExe   = Join-Path $ProjectDir "reachy_mini_env\Scripts\reachy-mini-daemon.exe"
$PythonExe   = Join-Path $ProjectDir "reachy_mini_env\Scripts\python.exe"
$AppScript   = Join-Path $ProjectDir "gesture_web_app.py"
$DaemonLog   = Join-Path $ProjectDir "daemon.log"
$AppLog      = Join-Path $ProjectDir "gesture_app.log"
$DaemonUrl   = "http://localhost:8000/docs"
$AppUrl      = "https://localhost:8765/status"

function Get-MatchingProcesses($pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='reachy-mini-daemon.exe'" |
        Where-Object { $_.CommandLine -like $pattern }
}

function Test-HttpUp($url, $insecure) {
    try {
        if ($insecure) {
            $resp = curl.exe -sk -o NUL -w "%{http_code}" $url --max-time 3
        } else {
            $resp = curl.exe -s -o NUL -w "%{http_code}" $url --max-time 3
        }
        return $resp -eq "200"
    } catch {
        return $false
    }
}

function Get-DaemonStatus {
    $procs = Get-MatchingProcesses "*reachy-mini-daemon*"
    $up = Test-HttpUp $DaemonUrl $false
    [PSCustomObject]@{
        Name      = "Daemon"
        Running   = [bool]$procs
        Responding = $up
        Pids      = ($procs | Select-Object -ExpandProperty ProcessId) -join ", "
    }
}

function Get-AppStatus {
    $procs = Get-MatchingProcesses "*gesture_web_app.py*"
    $up = Test-HttpUp $AppUrl $true
    $detail = $null
    if ($up) {
        try { $detail = curl.exe -sk $AppUrl --max-time 3 | ConvertFrom-Json } catch {}
    }
    [PSCustomObject]@{
        Name      = "App"
        Running   = [bool]$procs
        Responding = $up
        Pids      = ($procs | Select-Object -ExpandProperty ProcessId) -join ", "
        Detail    = $detail
    }
}

function Show-Status {
    $d = Get-DaemonStatus
    $a = Get-AppStatus

    Write-Host ""
    Write-Host "Daemon : " -NoNewline
    if ($d.Responding) { Write-Host "UP" -ForegroundColor Green -NoNewline }
    elseif ($d.Running) { Write-Host "RUNNING BUT NOT RESPONDING" -ForegroundColor Yellow -NoNewline }
    else { Write-Host "DOWN" -ForegroundColor Red -NoNewline }
    Write-Host "  (pid: $($d.Pids))"

    Write-Host "App    : " -NoNewline
    if ($a.Responding) { Write-Host "UP" -ForegroundColor Green -NoNewline }
    elseif ($a.Running) { Write-Host "RUNNING BUT NOT RESPONDING" -ForegroundColor Yellow -NoNewline }
    else { Write-Host "DOWN" -ForegroundColor Red -NoNewline }
    Write-Host "  (pid: $($a.Pids))"

    if ($a.Detail) {
        Write-Host ""
        Write-Host "  gesture         : $($a.Detail.gesture)"
        Write-Host "  mode            : $($a.Detail.mode)"
        Write-Host "  hands_detected  : $($a.Detail.hands_detected)"
        Write-Host "  fps             : $($a.Detail.fps)"
        Write-Host "  last_error      : $($a.Detail.last_error)"
        Write-Host "  phone_connected : $($a.Detail.phone_connected)"
        Write-Host "  phone_url       : $($a.Detail.phone_url)"
        Write-Host "  suppressed      : $($a.Detail.suppressed)"
        Write-Host ""
        Write-Host "  Desktop: https://localhost:8765"
    }
    Write-Host ""
}

function Stop-App {
    $procs = Get-MatchingProcesses "*gesture_web_app.py*"
    if (-not $procs) { Write-Host "App: not running."; return }
    foreach ($p in $procs) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "App: stopped pid $($p.ProcessId)."
    }
}

function Stop-Daemon {
    $procs = Get-MatchingProcesses "*reachy-mini-daemon*"
    if (-not $procs) { Write-Host "Daemon: not running."; return }
    foreach ($p in $procs) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Daemon: stopped pid $($p.ProcessId)."
    }
}

function Stop-All {
    Stop-App
    Stop-Daemon
    Start-Sleep -Seconds 1
}

function Wait-ForHttp($url, $insecure, $label, $timeoutSec = 30) {
    $elapsed = 0
    while ($elapsed -lt $timeoutSec) {
        if (Test-HttpUp $url $insecure) {
            Write-Host "${label}: up after ${elapsed}s."
            return $true
        }
        Start-Sleep -Seconds 1
        $elapsed++
    }
    Write-Host "${label}: did NOT come up within ${timeoutSec}s - check its log." -ForegroundColor Red
    return $false
}

function Start-Daemon {
    $d = Get-DaemonStatus
    if ($d.Responding) { Write-Host "Daemon: already up."; return }
    if ($d.Running -and -not $d.Responding) {
        Write-Host "Daemon: process present but not responding - restarting it."
        Stop-Daemon
        Start-Sleep -Seconds 1
    }
    Write-Host "Daemon: starting..."
    Start-Process -FilePath $DaemonExe -ArgumentList "--headless" `
        -RedirectStandardOutput $DaemonLog -RedirectStandardError "$DaemonLog.err" `
        -WorkingDirectory $ProjectDir -WindowStyle Hidden
    Wait-ForHttp $DaemonUrl $false "Daemon" 30 | Out-Null
}

function Start-App {
    $a = Get-AppStatus
    if ($a.Responding) { Write-Host "App: already up."; return }
    if ($a.Running -and -not $a.Responding) {
        Write-Host "App: process present but not responding - restarting it."
        Stop-App
        Start-Sleep -Seconds 1
    }
    Write-Host "App: starting..."
    Start-Process -FilePath $PythonExe -ArgumentList "`"$AppScript`"" `
        -RedirectStandardOutput $AppLog -RedirectStandardError "$AppLog.err" `
        -WorkingDirectory $ProjectDir -WindowStyle Hidden
    Wait-ForHttp $AppUrl $true "App" 30 | Out-Null
}

function Start-All {
    Start-Daemon
    Start-App
    Show-Status
}

function Show-Menu {
    while ($true) {
        Write-Host ""
        Write-Host "=== Reachy Mini - Manage ===" -ForegroundColor Cyan
        Write-Host "1) Status"
        Write-Host "2) Start"
        Write-Host "3) Stop"
        Write-Host "4) Restart"
        Write-Host "5) Exit"
        $choice = Read-Host "Choose an option (1-5)"
        switch ($choice) {
            "1" { Show-Status }
            "2" { Start-All }
            "3" { Stop-All }
            "4" { Stop-All; Start-All }
            "5" { return }
            default { Write-Host "Invalid choice - enter 1-5." -ForegroundColor Yellow }
        }
    }
}

switch ($Action) {
    "status"  { Show-Status }
    "start"   { Start-All }
    "stop"    { Stop-All }
    "restart" { Stop-All; Start-All }
    "menu"    { Show-Menu }
}

exit 0

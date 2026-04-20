# SignFlow AI — Management Script
# Usage:  .\manage.ps1 setup   (first time)
#         .\manage.ps1 run     (every time)

param (
    [Parameter(Mandatory=$false)]
    [String]$Action = "run"
)

$RootPath = Get-Location

# ── Locate a Python <=3.12 binary (required for TensorFlow) ──────────────────
function Get-Python312 {
    # Prefer explicit py -3.12
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        $ver = (py -3.12 --version 2>&1)
        if ($ver -match "Python 3\.12") {
            return @{ Exe = "py"; ExtraArgs = @("-3.12") }
        }
    }
    # Fallback: check if default 'python' is <=3.12
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $raw = (python --version 2>&1) -replace "Python ",""
        if ($raw -notmatch "not found") {
            $parts = $raw.Split(".")
            if ([int]$parts[0] -eq 3 -and [int]$parts[1] -le 12) {
                return @{ Exe = "python"; ExtraArgs = @() }
            }
        }
    }
    return $null
}

# Run any Python command through the located interpreter
function Invoke-Python312 {
    param($PyInfo, [string[]]$PyArgs)
    & $PyInfo.Exe @($PyInfo.ExtraArgs + $PyArgs)
}

# ── SETUP ────────────────────────────────────────────────────────────────────
function Setup-Environment {
    Write-Host "--- Discovering Python 3.12 (TensorFlow requires <=3.12) ---" -ForegroundColor Cyan
    $py = Get-Python312
    if ($null -eq $py) {
        Write-Host "ERROR: Python 3.12 not found." -ForegroundColor Red
        Write-Host "  Run: py install 3.12    then retry: .\manage.ps1 setup" -ForegroundColor Yellow
        return
    }
    Write-Host "Using: $($py.Exe) $($py.ExtraArgs -join ' ')" -ForegroundColor Green

    # --- Backend ---
    Write-Host "--- Setting up Backend ---" -ForegroundColor Cyan
    Set-Location "$RootPath\backend"

    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment with Python 3.12..."
        Invoke-Python312 $py @("-m", "venv", "venv")
    }

    Write-Host "Installing backend dependencies..."
    $venvPy = ".\venv\Scripts\python.exe"
    & $venvPy -m pip install --upgrade pip --quiet
    & $venvPy -m pip install fastapi uvicorn numpy python-multipart tensorflow --timeout 300

    # --- Frontend ---
    Write-Host "--- Setting up Frontend ---" -ForegroundColor Cyan
    Set-Location "$RootPath\frontend"
    Write-Host "Installing frontend dependencies..."
    npm install

    Set-Location $RootPath
    Write-Host "--- Setup Complete! Run '.\manage.ps1 run' to start. ---" -ForegroundColor Green
}

# ── RUN ──────────────────────────────────────────────────────────────────────
function Run-App {
    $BackendVenv  = "$RootPath\backend\venv"
    $FrontendMods = "$RootPath\frontend\node_modules"

    if (-not (Test-Path $BackendVenv)) {
        Write-Host "ERROR: Backend venv not found. Run '.\manage.ps1 setup' first." -ForegroundColor Red
        return
    }
    if (-not (Test-Path $FrontendMods)) {
        Write-Host "Frontend node_modules missing - running npm install..." -ForegroundColor Yellow
        Set-Location "$RootPath\frontend"
        npm install
        Set-Location $RootPath
    }

    Write-Host "--- Starting SignFlow AI Stack ---" -ForegroundColor Cyan

    $backendCmd  = "cd '$RootPath\backend'; .\venv\Scripts\python.exe main.py"
    $frontendCmd = "cd '$RootPath\frontend'; npm run electron:dev"

    Write-Host "  Launching Backend on http://localhost:8000 ..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

    Write-Host "  Launching Desktop Application (Electron) ..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

    Write-Host "Services are starting. The SignFlow AI window will appear shortly." -ForegroundColor Yellow
}

# ── COLLECT ─────────────────────────────────────────────────────────────────
function Collect-Data {
    param([string[]]$Signs)
    $BackendVenv = "$RootPath\backend\venv"
    if (-not (Test-Path $BackendVenv)) {
        Write-Host "ERROR: Run '.\manage.ps1 setup' first." -ForegroundColor Red; return
    }
    $script = "$RootPath\backend\collect_data.py"
    if ($Signs.Count -gt 0) {
        Write-Host "Collecting data for: $Signs" -ForegroundColor Cyan
        & "$BackendVenv\Scripts\python.exe" $script @Signs
    } else {
        Write-Host "Collecting data for ALL actions in config.py ..." -ForegroundColor Cyan
        & "$BackendVenv\Scripts\python.exe" $script
    }
}

# ── TRAIN ────────────────────────────────────────────────────────────────────
function Train-Model {
    $BackendVenv = "$RootPath\backend\venv"
    if (-not (Test-Path $BackendVenv)) {
        Write-Host "ERROR: Run '.\manage.ps1 setup' first." -ForegroundColor Red; return
    }
    Write-Host "--- Training LSTM model + converting to ONNX ---" -ForegroundColor Cyan
    Set-Location "$RootPath\backend"
    & "$BackendVenv\Scripts\python.exe" "train_model.py"
    Set-Location $RootPath
    Write-Host "Done! Restart the backend: .\manage.ps1 run" -ForegroundColor Green
}

# ── DISPATCH ─────────────────────────────────────────────────────────────────
$remainingArgs = $args   # extra args after $Action (e.g. sign names)

switch ($Action.ToLower()) {
    "setup"   { Setup-Environment }
    "run"     { Run-App }
    "collect" { Collect-Data -Signs $remainingArgs }
    "train"   { Train-Model }
    default   { 
        Write-Host "Unknown action '$Action'." -ForegroundColor Red
        Write-Host "  Available: setup  run  collect [signs...]  train" -ForegroundColor Yellow
    }
}


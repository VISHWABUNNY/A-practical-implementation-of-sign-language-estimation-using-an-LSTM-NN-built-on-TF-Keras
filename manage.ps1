# SignFlow AI Management Script

param (
    [Parameter(Mandatory=$false)]
    [String]$Action = "run"
)

$RootPath = Get-Location

function Setup-Environment {
    Write-Host "--- Discovering Python ---" -ForegroundColor Cyan
    $PythonCmd = ""
    
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $testPos = (python --version 2>&1)
        if ($testPos -notmatch "Python was not found") { $PythonCmd = "python" }
    }
    
    if ($PythonCmd -eq "" -and (Get-Command "py" -ErrorAction SilentlyContinue)) {
        $PythonCmd = "py"
    }

    if ($PythonCmd -eq "") {
        Write-Host "Error: Python was not found on your system." -ForegroundColor Red
        Write-Host "Please install Python from https://www.python.org/ and ensure it's in your PATH." -ForegroundColor Yellow
        return
    }
    
    Write-Host "Using Python command: $PythonCmd" -ForegroundColor Green

    Write-Host "--- Setting up Backend ---" -ForegroundColor Cyan
    Set-Location "$RootPath/backend"
    if (-not (Test-Path "venv")) {
        Write-Host "Creating Virtual Environment..."
        & $PythonCmd -m venv venv
    }
    Write-Host "Installing Backend Dependencies..."
    & $PythonCmd -m pip install --upgrade pip
    Get-Content requirements.txt | ForEach-Object {
        if ($_ -and -not $_.StartsWith("#")) {
            Write-Host "Installing $_..."
            & $PythonCmd -m pip install $_
        }
    }

    Write-Host "--- Setting up Frontend ---" -ForegroundColor Cyan
    Set-Location "$RootPath/frontend"
    Write-Host "Installing Frontend Dependencies..."
    npm install

    Set-Location $RootPath
    Write-Host "--- Setup Complete ---" -ForegroundColor Green
}

function Run-App {
    $BackendVenv = "$RootPath\backend\venv"
    $FrontendModules = "$RootPath\frontend\node_modules"

    if (-not (Test-Path $BackendVenv)) {
        Write-Host "Error: Backend virtual environment not found in $BackendVenv" -ForegroundColor Red
        Write-Host "Please run '.\manage.ps1 setup' first to initialize the environment." -ForegroundColor Yellow
        return
    }

    if (-not (Test-Path $FrontendModules)) {
        Write-Host "Info: Frontend dependencies (node_modules) not found." -ForegroundColor Yellow
        Write-Host "Running 'npm install' in frontend directory..." -ForegroundColor Cyan
        Set-Location "$RootPath\frontend"
        npm install
        Set-Location $RootPath
    }

    Write-Host "--- Starting SignFlow AI Stack ---" -ForegroundColor Cyan
    
    # Start Backend in a new window
    Write-Host "Launching Backend on http://localhost:8000..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; .\venv\Scripts\python.exe main.py"
    
    # Start Frontend Desktop App in a new window
    Write-Host "Launching Desktop Application (Electron)..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run electron:dev"

    Write-Host "Services are starting. The SignFlow AI window will appear shortly." -ForegroundColor Yellow
}

switch ($Action) {
    "setup" { Setup-Environment }
    "run"   { Run-App }
    default { 
        Write-Host "Unknown action: $Action. Use 'setup' or 'run'." -ForegroundColor Red
    }
}

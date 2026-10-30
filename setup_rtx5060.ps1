# 1-Click Automated Setup for RTX 5060 (8GB VRAM) on Windows
Write-Host "[+] Setting up AETHER3D Environment for RTX 5060..." -ForegroundColor Cyan

if (-not (Test-Path "C:\COLMAP\COLMAP.bat")) {
    Write-Host "[+] Downloading COLMAP CUDA for Windows to C:\COLMAP..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "C:\COLMAP" | Out-Null
    Invoke-WebRequest -Uri "https://github.com/colmap/colmap/releases/download/3.9.1/COLMAP-3.9.1-windows-cuda.zip" -OutFile "$env:TEMP\colmap.zip"
    Expand-Archive -Path "$env:TEMP\colmap.zip" -DestinationPath "C:\COLMAP" -Force
    Get-ChildItem -Path "C:\COLMAP\*" -Directory | ForEach-Object { Move-Item -Path "$($_.FullName)\*" -Destination "C:\COLMAP" -Force }
}

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

$pythonPath = (Get-Command python).Source
$regPath = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
if (-not (Test-Path $regPath)) { New-Item -Path $regPath -Force | Out-Null }
Set-ItemProperty -Path $regPath -Name $pythonPath -Value "GpuPreference=2;" -Force
Set-ItemProperty -Path $regPath -Name "C:\COLMAP\colmap.exe" -Value "GpuPreference=2;" -Force
Set-ItemProperty -Path $regPath -Name "C:\COLMAP\COLMAP.bat" -Value "GpuPreference=2;" -Force

python pre_demo_check.py

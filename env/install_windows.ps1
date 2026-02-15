$ErrorActionPreference = "Stop"

$ENV_FILE = "conda_windows.yml"
$REQ_FILE = "requirements_windows.txt"

Write-Host "===================================="
Write-Host "   MDock Environment Installer"
Write-Host "===================================="

# 检查 conda 是否存在
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
}

Write-Host "Creating conda environment..."
conda env create -f $ENV_FILE

# 自动读取环境名
$envName = (Select-String -Path $ENV_FILE -Pattern '^name:').ToString().Split()[1]

Write-Host "Activating environment: $envName"

# 关键：在当前 shell 中初始化 conda
& conda shell.powershell hook | Out-String | Invoke-Expression
conda activate $envName

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# 安装 pip 依赖
if (Test-Path $REQ_FILE) {
    Write-Host "Installing pip dependencies..."
    pip install -r $REQ_FILE
}

Write-Host "===================================="
Write-Host " Installation Complete!"
Write-Host "===================================="
Write-Host "Activate environment using:"
Write-Host "conda activate $envName"
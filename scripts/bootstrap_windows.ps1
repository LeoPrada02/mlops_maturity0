Write-Host "==> Creando ambiente conda mlops-local-demo"
conda env create -f environment.yml
Write-Host "==> Activando ambiente"
conda activate mlops-local-demo

Write-Host "==> Inicializando Git si hace falta"
if (-not (Test-Path ".git")) {
    git init
    git add .
    git commit -m "Initial local MLOps demo" 2>$null
}

Write-Host "==> Inicializando DVC si hace falta"
if (-not (Test-Path ".dvc")) {
    dvc init --force
    dvc remote add -d local_storage .dvc_storage
    git add .
    git commit -m "Initialize DVC" 2>$null
}

Write-Host "==> Bootstrap finalizado"

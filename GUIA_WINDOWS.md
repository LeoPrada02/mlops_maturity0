# Guía paso a paso para Windows 11 + Anaconda + VS Code

## 0. Prerrequisitos

Antes de empezar, verifica que tengas instalado:

- Anaconda o Miniconda
- Git
- VS Code
- Docker Desktop
- PowerShell

Además, abre Docker Desktop y confirma que esté corriendo correctamente.

---

## 1. Descomprimir el ZIP

1. Descarga el archivo ZIP.
2. Descomprímelo, por ejemplo en:
   `C:\proyectos\mlops_local_churn_demo`
3. Abre esa carpeta en VS Code.

---

## 2. Crear el ambiente Conda

Abre **Anaconda Prompt** dentro de la carpeta del proyecto y ejecuta:

```bash
conda env create -f environment.yml
conda activate mlops-local-demo
```

### Qué está pasando aquí
- `conda env create` crea un ambiente aislado.
- `conda activate` selecciona ese ambiente para que todo el proyecto use las mismas librerías.

---

## 3. Inicializar Git y DVC

Si el ZIP no conserva metadata de Git o DVC en tu equipo, ejecuta:

```bash
git init
git add .
git commit -m "Initial commit"

dvc init --force
dvc remote add -d local_storage .dvc_storage
```

### Qué está pasando aquí
- Git empieza a versionar el código.
- DVC prepara el proyecto para versionar pipeline y artefactos reproducibles.

---

## 4. Ejecutar la preparación y entrenamiento

```bash
python -m src.data.make_dataset
python -m src.models.train
```

O con DVC:

```bash
dvc repro
```

### Qué está pasando aquí
- `make_dataset` genera train/test/reference/scoring.
- `train` entrena el modelo, guarda artefactos y métricas.
- si MLflow está disponible y levantado, también registra la corrida y el modelo.

---

## 5. Ejecutar pruebas

```bash
pytest
```

### Qué está pasando aquí
- validas que exista configuración mínima y que la API exponga su contrato base.

---

## 6. Levantar el stack local completo

Con Docker Desktop abierto, ejecuta:

```bash
docker compose up --build
```

Esto levantará:

- MLflow en `http://localhost:5000`
- API FastAPI en `http://localhost:8000`
- Prometheus en `http://localhost:9090`
- Grafana en `http://localhost:3000`

### Qué está pasando aquí
- Docker construye la imagen de la API.
- Compose conecta todos los servicios.
- Prometheus empieza a scrapear `/metrics`.
- Grafana carga automáticamente el datasource y dashboard.

---

## 7. Probar la API

Puedes abrir:

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

También puedes hacer un POST a `/predict` con el archivo `data/scoring/sample_request_1.json`.

Ejemplo en PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -InFile "data/scoring/sample_request_1.json"
```

### Qué está pasando aquí
- el modelo responde una probabilidad de churn y una clase.
- la inferencia se registra en `monitoring/current_data.csv`.
- Prometheus acumula métricas de requests y latencia.

---

## 8. Generar tráfico simulado

```bash
python scripts/generate_traffic.py --n 100
```

Para forzar drift:

```bash
python scripts/generate_traffic.py --n 100 --drift
```

### Qué está pasando aquí
- se llaman muchas veces al endpoint `/predict`.
- el archivo `monitoring/current_data.csv` crece.
- con `--drift` se altera la distribución de inputs para simular una producción degradada.

---

## 9. Ejecutar Evidently

```bash
python -m src.monitoring.run_evidently
```

### Qué está pasando aquí
- se compara `data/reference/reference_data.csv` contra `monitoring/current_data.csv`.
- se genera:
  - `monitoring/reports/evidently_report.html`
  - `monitoring/reports/evidently_summary.json`

Abre el HTML en tu navegador.

---

## 10. Revisar observabilidad

### MLflow
Abre `http://localhost:5000`

Verás:
- runs,
- parámetros,
- métricas,
- artefactos,
- registro de modelo.

### Prometheus
Abre `http://localhost:9090`

Puedes probar queries como:
- `api_requests_total`
- `predictions_total`
- `prediction_latency_seconds_count`

### Grafana
Abre `http://localhost:3000`

Credenciales por defecto:
- usuario: `admin`
- contraseña: `admin`

Verás un dashboard provisionado automáticamente.

---

## 11. Detener todo

```bash
docker compose down
```

---

## Secuencia recomendada de aprendizaje

1. abrir notebook,
2. inspeccionar datos,
3. entrenar,
4. ver MLflow,
5. correr API,
6. generar tráfico,
7. revisar Prometheus/Grafana,
8. ejecutar Evidently.

Ese orden te ayuda a entender cada etapa del flujo MLOps.

# Demo MLOps local end-to-end: predicción de churn de clientes

Este proyecto implementa un flujo **MLOps completo en una computadora local** usando todas las herramientas clave del stack corto y práctico:

- Python
- conda
- VS Code
- JupyterLab
- pandas
- scikit-learn
- pytest
- Git
- DVC
- MLflow
- FastAPI
- Docker Desktop
- Docker Compose
- Prometheus
- Grafana
- Evidently

## Caso de uso

Se modela un problema simple pero útil: **predicción de churn de clientes** para una empresa de servicios de internet/telecom.

La variable objetivo (`churn`) indica si un cliente tiene alta probabilidad de abandonar el servicio.  
El dataset es **sintético**, fue incluido dentro del proyecto para que el flujo sea totalmente reproducible y funcione sin depender de descargas externas.

### Variables principales

- `tenure_months`: antigüedad del cliente
- `monthly_charge`: cargo mensual
- `total_charges`: monto acumulado
- `support_tickets`: tickets de soporte
- `late_payments`: pagos tardíos
- `avg_monthly_usage_gb`: uso promedio
- `contract_type`: mensual / anual / bianual
- `payment_method`: método de pago
- `internet_service`: tipo de servicio
- `has_streaming`: si tiene streaming
- `has_security_pack`: si tiene paquete de seguridad
- `num_products`: cantidad de productos
- `region`: región
- `customer_age`: edad
- `is_promo`: si ingresó por promoción

## Estructura

```text
mlops_local_churn_demo/
├── data/
├── artifacts/
├── monitoring/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── docker/
├── params.yaml
├── dvc.yaml
├── requirements.txt
├── environment.yml
└── docker-compose.yml
```

## Flujo del proyecto

1. **Datos**: dataset sintético ya incluido.
2. **Preparación**: split train/test y armado de datasets de referencia y scoring.
3. **Entrenamiento**: pipeline sklearn con preprocessing + logistic regression.
4. **Tracking**: métricas y registro en MLflow.
5. **Versionado**: pipeline reproducible en DVC y control de código con Git.
6. **Serving**: API de inferencia en FastAPI.
7. **Contenedores**: stack local en Docker Compose.
8. **Monitoreo técnico**: Prometheus + Grafana.
9. **Monitoreo de drift**: Evidently comparando producción vs referencia.
10. **Pruebas**: pytest sobre config y contrato básico de la API.

## Arranque rápido

Ver `GUIA_WINDOWS.md` para el paso a paso detallado.

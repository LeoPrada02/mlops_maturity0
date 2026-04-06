# Descripción detallada del caso de uso

## Problema de negocio

Una empresa de conectividad quiere anticipar qué clientes podrían abandonar el servicio en el corto plazo.  
La idea no es reemplazar un sistema productivo real, sino tener un ejemplo didáctico pero realista para practicar un flujo MLOps local completo.

## Objetivo analítico

Construir un modelo de clasificación binaria que estime la probabilidad de churn usando variables operativas y comerciales del cliente.

## Por qué este caso sirve para MLOps

Este caso es ideal para un demo local porque:

- es tabular y simple de ejecutar en una laptop,
- permite ilustrar **preparación, entrenamiento, evaluación, despliegue y monitoreo**,
- admite inferencia online vía API,
- permite simular **drift** alterando la distribución de algunos inputs.

## Qué hace cada herramienta en este caso

- **Python**: desarrollo general del pipeline.
- **conda**: ambiente aislado.
- **VS Code**: edición, debugging y pruebas.
- **JupyterLab**: exploración y entendimiento inicial de datos.
- **pandas**: manipulación de datos.
- **scikit-learn**: entrenamiento del modelo.
- **pytest**: validaciones automáticas.
- **Git**: versionado del código.
- **DVC**: versionado del pipeline y de los artefactos reproducibles.
- **MLflow**: tracking de runs y model registry local.
- **FastAPI**: serving del modelo como API.
- **Docker Desktop**: contenedores locales.
- **Docker Compose**: orquestación del stack.
- **Prometheus**: scraping de métricas.
- **Grafana**: dashboards de operación.
- **Evidently**: detección de drift entre referencia y scoring actual.

# Revisión del manuscrito de cebada NIR

Estado: cambios científicos y de formato incorporados; nombres, orden, afiliaciones, correos y ORCID proporcionados por el usuario e incorporados. Julián Coronel Reyes (jcoronel1@utmachala.edu.ec) es el autor de correspondencia designado por el usuario. URL GitHub, DOI archivado, licencia de software y aprobación final pendientes. No se ha enviado a revisores ni al congreso.

| Observación | Corrección y evidencia | Estado |
| --- | --- | --- |
| R1 M1: interpretación espectral | Métodos 4.7, resultados 5.5 y Figura 3, incrustada en Word. Ya estaba incrustada como Figura 2 en la versión anterior. VIP y coeficientes de cinco entrenamientos externos, perfiles adicionales exploratorios post-humedad, CSV por canal y bandas. La escala espectral es aproximada. VIP medio 1400–1500 nm: 0,775 en seis etapas y 0,909 post-humedad; no respalda predominio de esa banda. | Incorporada y auditada |
| R1 M2: relevancia post-humedad | Abstract, resultados, discusión y conclusión destacan MAE 0,680 y R² 0,618 en etapas 1–5. Agregado 0,577 y referencia seca 0,062; incluir la referencia reduce el MAE alrededor del 15%. Son las mismas predicciones originales, no un nuevo predictor post-only. | Incorporada y auditada |
| R1 M3: C en borde de grilla | Métodos 4.7, resultados 5.6 y Tabla 5. Comprobación congelada C=100/300/1000 con preprocesamiento, epsilon y regla gamma originales por fold. MAE agregado 0,577/0,590/0,619; no mejora con C mayor. No selección por errores externos ni sustitución de predicciones principales. | Incorporada y auditada |
| R2 M1: pequeño efecto de partición | Diferencia MAE fila–placa 0,021 presentada como resultado negativo útil; aporte centrado en benchmark auditable y dependencias. No es prueba de equivalencia ni efecto causal de fuga. | Incorporada |
| R2 M2: etapa y sesión | Abstract, introducción, discusión y conclusión dicen expresamente que no se puede descartar fingerprinting de sesión en vez de progreso biológico. VIP y agrupación por placa no resuelven el confundido. | Incorporada |
| R2 M3: transferencia | Calificador junto a Tabla 4: pruebas exploratorias a nivel de familia, no evidencia confirmatoria de robustez varietal. | Incorporada |
| R3 M1: depósito público con DOI | Paquete reproducible con código, notebook ejecutado, predicciones OOF, búsquedas por candidato, diagnósticos, dataset compacto y manifiesto SHA-256. Guía y metadatos Zenodo preparados; no DOI inventado. | Paquete verificado; publicación y DOI pendientes |
| R3 M2: completar manuscrito | Título, Abstract, nombres, afiliaciones, correos y ORCID incorporados. Julián Coronel Reyes es el autor de correspondencia. Por solicitud del usuario se retiran del artículo Author Contributions, Funding, Competing Interests y el párrafo AI assistance. Las declaraciones necesarias al envío requieren revisión de las instrucciones aplicables, sin afirmar ausencia de intereses o financiación. | Identificación completa; omisión de secciones solicitada por el usuario |
| R3 M3: licencia no comercial | Declaración compacta en metodología 4.6 y manifiesto de derechos: datos fuente y adaptaciones cubiertas conservan términos CC BY-NC 4.0. Los derechos del software son separados. | Incorporada; licencia de software requiere elección de titulares |
| Workflow metodológico | Figura 1 con validación anidada por placa, ajuste exclusivo en entrenamiento, ruta de test reservado, perfiles PLS, evaluación y diagnósticos exploratorios. PNG de 600 dpi y SVG editable; generador incluido en el paquete. | Incrustado en Word |
| Párrafo de GitHub | Disponibilidad concentrada en metodología 4.6. Único marcador solicitado: [GitHub repository URL to be inserted]. No se afirma que el repositorio sea público. | Redacción incorporada; URL pendiente |

## Verificaciones

- Auditoría primaria independiente aprobada; las predicciones principales permanecen sin cambios.
- Auditoría adicional independiente: 105 comprobaciones de métricas, 40.356 filas de predicción para tres C, 2.040 filas de perfiles PLS, identidad exacta C=100 con predicciones originales y membresía de folds comprobada.
- Tres pruebas unitarias de la fórmula VIP aprobadas. Sus fixtures matemáticos no son observaciones experimentales.
- Notebook ejecutado secuencialmente en modo lectura de resultados; no vuelve a entrenar por defecto.
- Word reconstruido con estilos y geometría de la plantilla Springer, con revisión visual de todas las páginas antes de entrega.

## Confirmaciones necesarias antes del envío

URL real del repositorio GitHub, publicación del código con licencia aprobada por sus titulares, aprobación del manuscrito por todos los autores y autorización del depósito público. Verificar las declaraciones exigibles en el envío: omitir sus bloques en el Word no confirma su contenido ni demuestra que ninguna declaración sea necesaria. El pago universitario de la inscripción comunicado por el usuario no equivale a una declaración de financiación de investigación.

GitHub no sustituye por sí solo el DOI de versión archivada pedido por el revisor R3 M1. Publicar un depósito aprobado en Zenodo y verificar su DOI sigue siendo una vía para satisfacerlo. Un ZIP privado o un DOI reservado sin publicación no cumple el pedido.

La asistencia de ChatGPT en código y redacción se conserva documentada en el README, fuera del manuscrito. No se añadieron datos experimentales sintéticos; los números se comprobaron contra predicciones guardadas. Antes de enviar, revisar cómo declarar esta asistencia conforme a la [política editorial de Springer](https://www.springernature.com/gp/policies/book-publishing-policies).

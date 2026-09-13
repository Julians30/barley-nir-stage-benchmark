# Preparación del depósito de reproducibilidad

El paquete contiene código, notebook, predicciones externas por observación, tablas de búsqueda internas, resultados de transferencia exploratoria, perfiles PLS, sensibilidad congelada de C y manifiestos SHA-256. El archivo local no equivale a un depósito público y no tiene un DOI inventado.

1. Confirmar SUBMISSION_METADATA_TEMPLATE.json: nombres y orden, afiliaciones, contribuciones, financiamiento, intereses y aprobación de todos los autores.
2. Revisar RIGHTS_AND_LICENSES.md y decidir la licencia del software. Conservar los términos CC BY-NC 4.0 de los componentes fuente y datos incorporados. Revisar si conviene depositar los espectros o enlazar únicamente su DOI ERDA.
3. Ingresar en la cuenta Zenodo de los autores y crear un registro en borrador. Completar los metadatos desde ZENODO_METADATA_TEMPLATE.json, que es una plantilla incompleta y no una petición API válida.
4. Cargar el paquete aprobado. Reservar un DOI si es útil para la edición, sin afirmar que está registrado mientras el registro siga en borrador.
5. Publicar solo con aprobación de los autores. Abrir el registro publicado, verificar sus archivos y copiar el DOI de la versión concreta.
6. Sustituir el campo de DOI pendiente en el manuscrito y registrar el identificador en SUBMISSION_METADATA_TEMPLATE.json. Confirmar que la cita apunta a esa versión antes del envío al congreso.

La corrección de reproducibilidad incluye preparación y verificación local. La publicación con DOI permanece pendiente hasta completar derechos y metadatos de autoría. No se enviaron documentos al congreso o a revisores.

Documentación oficial: https://help.zenodo.org/docs/deposit/create-new-upload/

## Recomposición del proyecto

Extraer el ZIP conservando las carpetas. Las rutas utilizadas por los scripts y el notebook son 01_DATASET, 02_CODIGO_NOTEBOOKS, 03_RESULTADOS, 04_MANUSCRITO_CCIS y 06_REFERENCIAS.

Lectura rápida: abrir 02_VALIDACION_ANIDADA_CEBADA_NIR.ipynb con los interruptores de recomputación en False.

Recomposición primaria opcional: ejecutar run_nested_validation.py con --project-dir apuntando a la raíz del proyecto. Los resultados compatibles ya terminados no se ajustan de nuevo.

Diagnósticos opcionales: ejecutar run_review_additions.py con --project-dir apuntando a la misma raíz, o el script principal con --review-additions. Sus archivos se guardan en 03_RESULTADOS/revision_r1 y no reemplazan las predicciones principales.

Reconstrucción de notebook y manuscrito: ejecutar build_final_artifacts.py desde la estructura de proyecto completa. El manuscrito usa estilos del archivo de referencia Springer y debe renderizarse e inspeccionarse después de cualquier edición.

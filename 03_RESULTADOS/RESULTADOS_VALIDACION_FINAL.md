# Resultados finales de validación — CITI 2027

## Diseño principal

- Validación cruzada anidada: 5 pliegues externos y 4 internos.
- Agrupación externa e interna: caja de Petri.
- Unidad del bootstrap: caja de Petri.
- Repeticiones bootstrap: 2,000.
- Variable objetivo: índice de etapa experimental (0–5), expresado en unidades diarias.
- Etapa 0: referencia seca previa a la humectación; etapas 1–5: adquisiciones diarias posteriores.

## Dificultad central: etapas post-humedad

Las mismas predicciones externas originales dan MAE 0,680, RMSE 0,875 y R² 0,618 en etapas 1–5. Incluir la referencia seca (MAE 0,062) reduce el MAE alrededor del 15%. No es un predictor nuevo ajustado solo a etapas post-humedad.

## Resultado agregado de seis etapas

La mejor familia fue **SVR_RBF**. La configuración de consenso fue **savgol_d1; C=100;epsilon=0.05;gamma=scale**.

- MAE: 0.577 días (IC 95% por bootstrap de placas: 0.560–0.594).
- RMSE: 0.800 días (IC 95%: 0.774–0.826).
- R²: 0.781 (IC 95%: 0.766–0.794).
- Predicciones dentro de ±1 día: 80.3% (IC 95%: 79.1–81.4%).

## Comparación de familias

| family | mae_mean | mae_sd | rmse_mean | rmse_sd | r2_mean | r2_sd | within_1_day_mean | bias_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVR_RBF | 0.5770 | 0.0102 | 0.7999 | 0.0127 | 0.7806 | 0.0070 | 0.8026 | -0.0153 |
| PLSR | 0.7218 | 0.0137 | 0.9006 | 0.0217 | 0.7218 | 0.0135 | 0.7246 | -0.0010 |
| ExtraTrees | 0.8558 | 0.0176 | 1.1295 | 0.0261 | 0.5624 | 0.0202 | 0.6169 | 0.0029 |

## Auditoría de esquemas de partición

Esta comparación usa una configuración fija elegida mediante la validación agrupada; es un análisis de sensibilidad y no reemplaza la estimación anidada principal.

| scheme | mae | rmse | r2 | within_1_day | bias |
| --- | --- | --- | --- | --- | --- |
| random_row | 0.5545 | 0.7637 | 0.8000 | 0.8144 | -0.0152 |
| grain_disjoint | 0.5632 | 0.7778 | 0.7925 | 0.8138 | -0.0170 |
| dish_disjoint | 0.5752 | 0.7964 | 0.7825 | 0.8040 | -0.0173 |

## Transferencia exploratoria

Las pruebas por cultivar y por grupo muestral son análisis de estrés. Con solo dos cultivares y cuatro grupos no deben interpretarse como una estimación poblacional amplia.

| scheme | held_out | family | preprocessing | parameter | mae | rmse | r2 | within_1_day | bias | n_train | n_test | n_train_dishes | n_test_dishes | fit_predict_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_cultivar_out | Laureate | SVR_RBF | savgol_d1 | C=100;epsilon=0.05;gamma=scale | 0.6670 | 0.8992 | 0.7228 | 0.7451 | 0.0077 | 6108 | 7344 | 41 | 49 | 11.9469 |
| leave_one_cultivar_out | Prospect | SVR_RBF | snv | C=100;epsilon=0.05;gamma=scale | 0.8153 | 1.1516 | 0.5453 | 0.6994 | -0.2849 | 7344 | 6108 | 49 | 41 | 11.0472 |
| leave_one_sample_group_out | laureate_0 | SVR_RBF | snv | C=100;epsilon=0.05;gamma=scale | 0.6670 | 0.8939 | 0.7260 | 0.7366 | 0.1722 | 9708 | 3744 | 65 | 25 | 16.4104 |
| leave_one_sample_group_out | laureate_1 | SVR_RBF | savgol_d1 | C=100;epsilon=0.05;gamma=scale | 0.6139 | 0.8249 | 0.7667 | 0.7758 | -0.1195 | 9852 | 3600 | 66 | 24 | 26.8698 |
| leave_one_sample_group_out | prospect_0 | SVR_RBF | savgol_d1 | C=100;epsilon=0.05;gamma=scale | 0.7893 | 1.0556 | 0.6180 | 0.6822 | 0.3908 | 9708 | 3744 | 65 | 25 | 23.6009 |
| leave_one_sample_group_out | prospect_1 | SVR_RBF | savgol_d1 | C=100;epsilon=0.05;gamma=scale | 0.8558 | 1.2129 | 0.4956 | 0.6730 | -0.5440 | 11088 | 2364 | 74 | 16 | 31.1696 |

## Nota de interpretación

El objetivo es estimar la etapa experimental de una adquisición. No hay observaciones intermedias que validen una resolución temporal subdiaria. No se predicen viabilidad, concentración de humedad ni germinación prospectiva.

Separar placas evita compartir granos o placas entre entrenamiento y test, pero no separa sesiones de adquisición. La etapa puede estar confundida con efectos de sesión o instrumento compartidos. No se demuestra transferencia a nuevas sesiones, cosechas o equipos. Las rejillas se orientaron mediante factibilidad sobre este mismo dataset.

Los intervalos bootstrap son condicionales a las predicciones externas guardadas: remuestrean placas, pero no reajustan modelos ni repiten la selección de familia. Las pruebas de transferencia son exploratorias a nivel de familia.

## Comprobación de etapas posteriores a la humectación

Se restringen las mismas predicciones externas a etapas 1–5, sin reajustar un modelo. La referencia constante se calcula sobre cada subconjunto. Es un análisis descriptivo, no una prueba independiente.

| subset | predictor | n | mae | rmse | r2 | within_1_day | bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_six_stages | SVR_RBF | 13452 | 0.5770 | 0.7999 | 0.7806 | 0.8026 | -0.0153 |
| all_six_stages | constant_subset_mean | 13452 | 1.5000 | 1.7078 | 0.0000 | 0.3333 | 0.0000 |
| post_moisture_1_to_5 | SVR_RBF | 11210 | 0.6800 | 0.8745 | 0.6176 | 0.7634 | -0.0196 |
| post_moisture_1_to_5 | constant_subset_mean | 11210 | 1.2000 | 1.4142 | 0.0000 | 0.6000 | 0.0000 |

## Diagnósticos incorporados en la revisión

La diferencia MAE fila–placa es pequeña (0,021): resultado negativo útil, no demostración de gran inflación por fuga ni equivalencia. Etapa y sesión coinciden en una secuencia; no se puede descartar fingerprinting de sesión en vez de progreso biológico. Las pruebas de transferencia son exploratorias a nivel de familia.

VIP y coeficientes de PLS se calcularon únicamente en entrenamientos externos. El VIP medio de la ventana aproximada 1400–1500 nm es inferior a uno y no respalda que esa banda domine PLS. El eje espectral es aproximado; estos perfiles no explican SVR directamente ni validan concentración química de humedad.

Sensibilidad congelada de C, sin sustituir predicciones principales o seleccionar por errores externos:

| C | subset | n | mae | rmse | r2 | within_1_day | bias | all_stage_fold_mae_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | all_six_stages | 13452 | 0.5770 | 0.7999 | 0.7806 | 0.8026 | -0.0153 | 0.0102 |
| 100 | post_moisture_1_to_5 | 11210 | 0.6800 | 0.8745 | 0.6176 | 0.7634 | -0.0196 | 0.0102 |
| 300 | all_six_stages | 13452 | 0.5898 | 0.8227 | 0.7679 | 0.8010 | -0.0054 | 0.0126 |
| 300 | post_moisture_1_to_5 | 11210 | 0.6917 | 0.8980 | 0.5968 | 0.7617 | -0.0081 | 0.0126 |
| 1000 | all_six_stages | 13452 | 0.6194 | 0.8872 | 0.7301 | 0.7931 | -0.0008 | 0.0206 |
| 1000 | post_moisture_1_to_5 | 11210 | 0.7213 | 0.9668 | 0.5326 | 0.7525 | -0.0028 | 0.0206 |

Código, notebook y resultados están preparados para depósito reproducible, pero no se ha publicado un registro Zenodo ni registrado un DOI. Autor de correspondencia designado por el usuario: Julián Coronel Reyes (jcoronel1@utmachala.edu.ec). GitHub permanece sin URL; no sustituye el DOI de versión archivada solicitado por el revisor. Por petición del usuario, las secciones de declaraciones no aparecen en el manuscrito; su omisión no confirma ausencia de financiación o intereses. La aprobación final, las declaraciones exigibles al envío y la licencia del software requieren confirmación. El workflow es la Figura 1, las predicciones la Figura 2 y los perfiles espectrales la Figura 3; las tres imágenes están incrustadas en Word.

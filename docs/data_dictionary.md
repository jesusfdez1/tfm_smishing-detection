# Diccionario de Datos del Meta-Dataset SMS

Este documento sirve como "biblia" para entender todas las columnas, categorías y valores posibles que conforman el meta-dataset final `metasms_hss_master.csv`. Toda la información contenida en el dataset obedece a las siguientes reglas y diccionarios de datos.

## Estructura General de las Columnas

| Columna | Descripción | Tipo de Dato / Valores Posibles |
|---------|-------------|--------------------------------|
| **`message_id`** | Identificador único del mensaje (ej. `msg_1a2b3c4d`). | `String (Hash)` |
| **`text`** | Texto original del mensaje (antes de anonimizar, si procede). | `String` |
| **`text_anonymized`** | Texto procesado ocultando PII (Sustituyendo URLs, teléfonos). | `String` |
| **`anonymization_status`** | Estado de anonimización del texto. | `Categoría` (Ver sección 2) |
| **`canonical_label`** | Etiqueta unificada definitiva. | `Categoría` (Ver sección 3) |
| **`timestamp_original`** | Fecha y hora original del mensaje (si estaba disponible). | `String (ISO 8601)` o Vacío |
| **`source`** | Nombre del dataset o fuente de donde proviene. | `Categoría` (Ver sección 4) |
| **`source_id`** | ID original del mensaje en su dataset de origen. | `String` o Vacío |
| **`source_url`** | URL de donde se extrajo el dataset. | `String (URL)` |
| **`original_label`** | La etiqueta original que tenía el mensaje en su origen. | `String` |
| **`label_mapping_rule`** | Regla aplicada para transformar la etiqueta original. | `String` |
| **`license`** | Licencia de uso del mensaje/dataset original. | `String` |
| **`language`** | Idioma principal detectado. | `Categoría` (Ver sección 5) |
| **`language_confidence`** | Nivel de confianza de la detección del idioma. | `Float (0.0 - 1.0)` |
| **`is_ai_generated`** | Indica si el mensaje ha sido generado por IA de forma sintética. | `Booleano (0 o 1)` |
| **`generation_model`** | Nombre del modelo que generó el mensaje sintético. | `String` (ej. `gpt-4`) o Vacío |
| **`label_inferred_by_ai`** | Indica si la etiqueta final (`canonical_label`) fue inferida por el LLM. | `Booleano (0 o 1)` |
| **`llm_annotated`** | Indica si el mensaje fue enriquecido por un LLM. | `Booleano (0 o 1)` |
| **`llm_model`** | El modelo usado para enriquecer la fila. | `String` (ej. `gemini-3.1-pro`) |
| **`prompt_version`** | La versión del prompt que se utilizó en el LLM. | `String` |
| **`llm_annotation_date`** | Fecha en la que el LLM procesó esta fila. | `String (ISO 8601)` |
| **`theme`** | Categoría o intención principal del mensaje. | `Categoría` (Ver sección 6) |
| **`urgency_level`** | Nivel de urgencia que transmite el mensaje. | `Categoría` (Ver sección 7) |

---

## Diccionarios de Valores y Categorías

### 1. Variables Booleanas
Varias columnas utilizan formato booleano estandarizado con valores enteros (0 y 1) para facilitar su procesamiento en Machine Learning:
- **`0`**: Falso (No aplica)
- **`1`**: Verdadero (Sí aplica)

*Aplica a las columnas: `is_ai_generated`, `label_inferred_by_ai`, y `llm_annotated`.*

### 2. `anonymization_status`
Define qué tipo de procesamiento de privacidad ha recibido el mensaje:
- **`raw`**: El mensaje no ha sufrido ningún cambio (no contiene información personal identificable o no se le ha aplicado anonimización).
- **`anonymized`**: El mensaje contenía PII y ha sido anonimizado por nuestro script (ej. reemplazo de enlaces por `[URL]`).
- **`pre_anonymized`**: El mensaje ya venía con tokens de anonimización aplicados en el dataset de origen (ej. `<URL>`).

### 3. `canonical_label` (Clasificación Principal)
Es la etiqueta unificada que deben predecir los modelos de Machine Learning (la variable objetivo dependiente):
- **`ham`**: Mensaje legítimo, inofensivo y seguro (ej. confirmación de compra real, mensajes entre familiares).
- **`spam`**: Mensaje promocional comercial no deseado pero no malicioso, ni delictivo ni fraudulento (ej. publicidad agresiva de seguros).
- **`smishing`**: Mensaje fraudulento o malicioso (phishing por SMS) diseñado para robar credenciales, engañar económicamente o infectar el dispositivo.

### 4. `source` (Fuentes de Origen)
Indica de qué dataset original proviene la fila. Valores posibles extraídos en el pipeline ETL:
- `agarwal_2025`
- `exais_sms`
- `hosseinpour_2025`
- `kaggle_phishing`
- `kaggle_spam_ham`
- `malicious_benign_sms_mms`
- `malicious_benign_synthetic`
- `mimics_3500`
- `mishra_extended`
- `mishra_soni_2022`
- `nus_sms`
- `smishing_4c`
- `smishtank`
- `spanish_spam_ham`
- `uci_sms_spam`

### 5. `language` (Idiomas ISO 639-1)
Los idiomas se estandarizan utilizando la norma internacional ISO 639-1 (dos letras minúsculas). Ejemplos más comunes en nuestro dataset:
- `en` (Inglés)
- `es` (Español)
- `fr` (Francés)
- `pt` (Portugués)
- `it` (Italiano)
- `de` (Alemán)
- `nl` (Neerlandés)
- `unknown` (No se ha podido detectar, texto ininteligible, demasiado corto o nulo)

### 6. `theme` (Intenciones y Temáticas)
Clasifica el propósito o temática central del mensaje. El LLM encasilla obligatoriamente el mensaje en una sola de estas categorías para todo el dataset:
- **Accounts**: Problemas con cuentas de usuario, verificación de identidad, códigos PIN, bloqueos y alertas de seguridad (no puramente bancarias).
- **Bank/Finance**: Banca corporativa, finanzas, préstamos, hipotecas, tarjetas de crédito y avisos de cargos.
- **Customer service**: Servicio de atención al cliente general (no bancario ni de envíos).
- **Dating**: Aplicaciones de citas, encuentros, contactos personales o mensajes románticos.
- **Deliveries**: Paquetería, aduanas, seguimiento de envíos (Correos, FedEx, UPS, Amazon).
- **Gifts**: Regalos gratuitos directos, donaciones.
- **Offers**: Ofertas comerciales, descuentos, cupones, rebajas promocionales (marketing agresivo).
- **Prizes**: Premios ganados (sin haber participado), sorteos, loterías.
- **Rewards**: Programas de fidelidad, puntos acumulados de clientes, cashback.
- **SMS Service**: Mensajes de servicio de la operadora de telefonía, recordatorios, códigos OTP (One Time Password).
- **Sexual**: Contenido para adultos, webcam, pornografía explícita.
- **Spam**: Contenido basura promocional general no clasificable en los anteriores (ej. venta de gafas de sol chinas, remedios milagrosos).
- **Lifestyle**: Eventos de estilo de vida, clubes, gimnasios, ocio general.
- **Other**: Cualquier otro mensaje que sea imposible encajar lógicamente en ninguna de las categorías anteriores.

### 7. `urgency_level` (Nivel de Urgencia)
Representa la presión psicológica temporal que el mensaje intenta ejercer sobre el usuario, escalado del 0 al 3:
- **0 - Nulo (Informativo)**: Ninguna prisa. Un simple aviso pasivo. _Ej: "Su saldo final es de 10€"._
- **1 - Bajo**: Acción requerida, pero sin un marco temporal estricto ni amenazas. _Ej: "Recuerda actualizar tu perfil de usuario en nuestra web cuando puedas"._
- **2 - Medio**: Requiere atención en el corto plazo (próximos días) o hay un beneficio a punto de caducar. _Ej: "Su paquete llega mañana, confirme su dirección"._
- **3 - Alto**: Acción inmediata o crítica. Amenazas de bloqueo, suspensión de cuenta, multas o cargos inminentes. _Ej: "URGENTE: Cargo de 500€ retenido. Si no es suyo, cancele ahora en [URL]"._

---

## Ejemplo Ficticio Completo

Para ilustrar cómo se engrana todo este diccionario en una fila real del CSV final, aquí tienes un ejemplo que emula un mensaje fraudulento procesado por completo a través de las dos fases de nuestro pipeline:

| Columna | Valor de Ejemplo | Explicación basada en los Diccionarios |
|---------|------------------|----------------------------------------|
| `message_id` | `msg_8f9a2b1c4e5d` | Hash único generado por el script en la fase 1. |
| `text` | `Banco Santander: Su tarjeta ha sido limitada. Para reactivarla visite http://bit.ly/sant-phish` | Texto tal cual viene del dataset origen, sin modificar. |
| `text_anonymized` | `Banco Santander: Su tarjeta ha sido limitada. Para reactivarla visite [URL]` | La URL maliciosa ha sido sustituida por el token `[URL]`. |
| `anonymization_status` | `anonymized` | Nuestro script aplicó Regex en fase 1 para limpiar PII. |
| `canonical_label` | `smishing` | Categoría unificada (ataque de phishing vía SMS). |
| `timestamp_original` | | No constaba la hora original en el dataset de origen. |
| `source` | `smishtank` | Proviene del repositorio web de SmishTank. |
| `source_id` | `102394` | ID de la sumisión original en la base de datos de SmishTank. |
| `source_url` | `https://smishtank.com/` | URL original del recurso o dataset. |
| `original_label` | `unlabeled (raw submission)` | En el JSON original no venía etiquetado como validado por la comunidad. |
| `label_mapping_rule` | `unlabeled->smishing (LLM Inference)` | El LLM dedujo la etiqueta al carecer de una en la fase 2. |
| `license` | `Academic Use Only` | Condición de uso estipulada por la fuente original. |
| `language` | `es` | Idioma español ISO 639-1 (Detectado por `langdetect`). |
| `language_confidence` | `0.998` | 99.8% de confianza del modelo detector de lenguaje. |
| `is_ai_generated` | `0` | No se detectó origen sintético. Es un SMS real interceptado. |
| `generation_model` | | (Vacío al no ser un SMS creado por Inteligencia Artificial). |
| `label_inferred_by_ai` | `1` | Verdadero (1), la etiqueta final se dedujo vía API del LLM. |
| `llm_annotated` | `1` | Verdadero (1), la fila pasó por la tubería de enriquecimiento y contiene meta-datos. |
| `llm_model` | `gemini-3.1-pro` | Modelo de Google Gemini utilizado para enriquecer este mensaje. |
| `prompt_version` | `v2.1` | Versión interna del script que construye los prompts de anotación. |
| `llm_annotation_date` | `2026-05-28T21:50:00Z` | Fecha ISO en la que el LLM nos devolvió el resultado. |
| **`theme`** | `Bank/Finance` | Encaja exactamente en la categoría bancaria. |
| **`urgency_level`** | `3` | Nivel máximo de urgencia por amenaza explícita de limitación de tarjeta. |

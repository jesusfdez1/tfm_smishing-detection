# Diccionario de Datos del Meta-Dataset SMS

Este documento sirve como "biblia" para entender todas las columnas, categorías y valores posibles que conforman el meta-dataset final `metasms_hss_master.csv`. Toda la información contenida en el dataset obedece a las siguientes reglas y diccionarios de datos.

## Estructura General de las Columnas

| Columna | Descripción | Tipo de Dato / Valores Posibles |
|---------|-------------|--------------------------------|
| **`message_id`** | Identificador único del mensaje (ej. `msg_1a2b3c4d`). | `String (Hash)` |
| **`canonical_label`** | Etiqueta unificada definitiva. | `Categoría` (Ver sección 3) |
| **`text_anonymized`** | Texto procesado ocultando PII (Sustituyendo URLs, teléfonos). | `String` |
| **`text`** | Texto original del mensaje (antes de anonimizar, si procede). | `String` |
| **`reference`** | Nombre del dataset o fuente de donde proviene. | `Categoría` (Ver sección 4) |
| **`source_id`** | ID original del mensaje en su dataset de origen. | `String` o Vacío |
| **`original_label`** | La etiqueta original que tenía el mensaje en su origen. | `String` |
| **`label_mapping_rule`** | Regla aplicada para transformar la etiqueta original. | `String` |
| **`timestamp_original`** | Fecha y hora original del mensaje (si estaba disponible). | `String (ISO 8601)` o Vacío |
| **`license`** | Licencia de uso del mensaje/dataset original. | `String` |
| **`language`** | Idioma principal detectado. | `Categoría` (Ver sección 5) |
| **`language_confidence`** | Nivel de confianza de la detección del idioma. | `Float (0.0 - 1.0)` |
| **`anonymization_status`** | Estado de anonimización del texto. | `Categoría` (Ver sección 2) |
| **`llm_model`** | El modelo usado para enriquecer la fila. | `String` (ej. `gemini-3.1-pro`) |
| **`prompt_version`** | La versión del prompt que se utilizó en el LLM. | `String` |
| **`llm_annotation_date`** | Fecha en la que el LLM procesó esta fila. | `String (ISO 8601)` |
| **`theme`** | Categoría o intención principal del mensaje. | `Categoría` (Ver sección 6) |
| **`urgency_level`** | Nivel de urgencia que transmite el mensaje. | `Categoría` (Ver sección 7) |
| **`whois_domain`** | Dominio extraído del enlace (si existe) tras la resolución de Whois. | `String` |
| **`whois_tld`** | Dominio de nivel superior (Top Level Domain) extraído del dominio. | `String` |
| **`whois_age_days`** | Edad en días del dominio extraída mediante Whois. | `Integer` |
| **`whois_hidden`** | Indica si los detalles de Whois del dominio están ocultos/privados (1 o 0). | `Booleano (0 o 1)` |
| **`whois_country`** | País de registro del dominio obtenido vía Whois. | `String` |

---

## Diccionarios de Valores y Categorías

### 1. Variables Booleanas
Varias columnas utilizan formato booleano estandarizado con valores enteros (0 y 1) para facilitar su procesamiento en Machine Learning:
- **`0`**: Falso (No aplica)
- **`1`**: Verdadero (Sí aplica)

*Aplica a las columnas: `whois_hidden` (si se considera booleana).*

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

### 4. `reference` (Fuentes de Origen)
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
Clasifica el propósito o temática central del mensaje. El LLM encasilla obligatoriamente el mensaje en una sola de estas 11 categorías (taxonomía cerrada) para todo el dataset:
- **`personal`**: Conversación casual, saludos, preguntas entre personas (ej. "What's up?").
- **`banking`**: Bancos, transferencias, tarjetas de crédito, criptomonedas.
- **`delivery`**: Paquetería, correos, aduanas, seguimiento de envíos.
- **`account_security`**: Bloqueo de cuentas, inicios de sesión sospechosos, verificación de identidad.
- **`promotion`**: Premios, loterías, descuentos, ofertas comerciales.
- **`government`**: Multas de tráfico (DGT), impuestos (Hacienda), notificaciones de la administración pública.
- **`job_offer`**: Ofertas de reclutamiento, empleo falso, ganar dinero fácil.
- **`dating_adult`**: Citas, contenido sexual, contactos.
- **`subscription`**: Servicios SMS premium, horóscopos, ringtones.
- **`service_alert`**: Citas médicas, cortes de suministro (luz/agua), recordatorios de servicios.
- **`unknown`**: No se puede determinar o no tiene sentido.

### 7. `urgency_level` (Nivel de Urgencia)
Representa la presión psicológica temporal que el mensaje intenta ejercer sobre el usuario, utilizando una taxonomía cerrada de 4 niveles:
- **`none`**: Sin urgencia, charla normal o informativa.
- **`low`**: Informativo, requiere acción pero sin presión de tiempo.
- **`medium`**: Requiere atención pronto (ej. "responde cuando puedas", "tu paquete llegará mañana").
- **`high`**: Acción inmediata requerida. Usa lenguaje de amenaza, miedo o límites de tiempo muy cortos (ej. "tu cuenta será bloqueada en 1 hora", "haz clic ya o perderás el premio").

---

## Ejemplo Ficticio Completo

Para ilustrar cómo se engrana todo este diccionario en una fila real del CSV final, aquí tienes un ejemplo que emula un mensaje fraudulento procesado por completo a través de las dos fases de nuestro pipeline:

| Columna | Valor de Ejemplo | Explicación basada en los Diccionarios |
|---------|------------------|----------------------------------------|
| `message_id` | `msg_8f9a2b1c4e5d` | Hash único generado por el script en la fase 1. |
| `canonical_label` | `smishing` | Categoría unificada (ataque de phishing vía SMS). |
| `text_anonymized` | `Banco Santander: Su tarjeta ha sido limitada. Para reactivarla visite [URL]` | La URL maliciosa ha sido sustituida por el token `[URL]`. |
| `text` | `Banco Santander: Su tarjeta ha sido limitada. Para reactivarla visite http://bit.ly/sant-phish` | Texto tal cual viene del dataset origen, sin modificar. |
| `reference` | `smishtank` | Proviene del repositorio web de SmishTank. |
| `source_id` | `102394` | ID de la sumisión original en la base de datos de SmishTank. |
| `original_label` | `unlabeled (raw submission)` | En el JSON original no venía etiquetado como validado por la comunidad. |
| `label_mapping_rule` | `unlabeled->smishing (LLM Inference)` | El LLM dedujo la etiqueta al carecer de una en la fase 2. |
| `timestamp_original` | | No constaba la hora original en el dataset de origen. |
| `license` | `Academic Use Only` | Condición de uso estipulada por la fuente original. |
| `language` | `es` | Idioma español ISO 639-1 (Detectado por `langdetect`). |
| `language_confidence` | `0.998` | 99.8% de confianza del modelo detector de lenguaje. |
| `anonymization_status` | `anonymized` | Nuestro script aplicó Regex en fase 1 para limpiar PII. |
| `llm_model` | `gemini-3.1-pro` | Modelo de Google Gemini utilizado para enriquecer este mensaje. |
| `prompt_version` | `v3.0` | Versión interna del script que construye los prompts de anotación. |
| `llm_annotation_date` | `2026-05-28T21:50:00Z` | Fecha ISO en la que el LLM nos devolvió el resultado. |
| **`theme`** | `banking` | Encaja exactamente en la taxonomía bancaria. |
| **`urgency_level`** | `high` | Nivel máximo de urgencia por amenaza explícita de limitación de tarjeta. |
| **`whois_domain`** | `bit.ly` | Dominio extraído de la URL. |
| **`whois_tld`** | `ly` | TLD del dominio. |
| **`whois_age_days`** | `4500` | Edad del dominio en días. |
| **`whois_hidden`** | `0` | El Whois no está oculto. |
| **`whois_country`** | `US` | País del registrador. |

---

## Anexo: Prompt Maestro (LLM)

Para garantizar la reproducibilidad y la transparencia en la investigación (trazabilidad del campo `prompt_version`), a continuación se expone el prompt exacto de sistema (versión **`v3.0`**) utilizado para extraer las características `theme` y `urgency_level` mediante Zero/Few-Shot Learning con Chain-of-Thought (CoT):

```text
You are an expert cybersecurity analyst annotating SMS messages for a machine learning dataset.
You will receive a JSON array of text messages. The input messages may be in multiple languages (Spanish, English, French, etc.). Analyze the semantic meaning in its native language, but ALWAYS write your "reasoning" and JSON keys in English.

You must return a strict JSON array of objects, one for each input message, in the EXACT SAME ORDER.

For each object, you must include EXACTLY these keys:
- "reasoning": string, a brief 1-2 sentence explanation of your analysis in English.
- "theme": string, strictly one of the themes listed below.
- "urgency_level": string, strictly one of the urgency levels listed below.

Themes mapping (Strictly MECE):
- personal: Casual conversation, greetings, family matters, or personal questions.
- banking: Banks, wire transfers, credit/debit cards, crypto, or specific financial alerts.
- delivery: Packages, post office, customs fees, shipping tracking (e.g. Correos, DHL).
- account_security: Account locks, suspicious logins, PIN/OTP codes, or identity verification (even if related to a bank, if the focus is the account login/security, use this).
- promotion: Prizes won, lotteries, discounts, aggressive commercial offers, or gifts.
- government: Traffic fines (DGT), taxes (Hacienda), public administration notifications.
- job_offer: Recruitment, job offers, work from home opportunities, easy money scams.
- dating_adult: Dating, sexual content, "hot singles", adult contacts.
- subscription: Premium SMS services, horoscopes, ringtones, paid subscriptions.
- service_alert: Medical appointments, utility bills/outages, mobile carrier service reminders.
- unknown: Completely unreadable, lacks context, or does not fit anywhere else.

Urgency levels:
- none: No urgency at all. Normal chat or purely passive information.
- low: Informative, requires some future action but with absolutely no time pressure.
- medium: Requires attention soon (e.g., "reply when you can", "your package arrives tomorrow").
- high: Immediate action required. Heavy psychological pressure, threats of account suspension, fines, or very short time limits (e.g., "Act within 24h or lose your account").

EXAMPLES:

Input: ["Hey mom, can you pick me up at 5?", "OFERTA: 50% de descuento en tus gafas de sol. Compra ya en opticasol.es/baja", "URGENT: Your bank account is suspended. Verify your identity immediately at http://secure-bank-login.com"]
Output: [
  {"reasoning": "Personal communication between family members. No malicious intent or marketing.", "theme": "personal", "urgency_level": "none"},
  {"reasoning": "Marketing message from a commercial entity offering a discount. Not deceptive.", "theme": "promotion", "urgency_level": "medium"},
  {"reasoning": "High-urgency deceptive message attempting to steal credentials via a fake URL under the guise of account suspension.", "theme": "account_security", "urgency_level": "high"}
]

Respond ONLY with valid JSON. No markdown, no formatting.
```

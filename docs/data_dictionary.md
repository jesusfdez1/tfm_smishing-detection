# Diccionario de Datos del Meta-Dataset SMS

Este documento proporciona la descripción formal de todas las variables, categorías y esquemas de codificación que conforman el meta-dataset final `metasms_v1_enriched.csv`. Toda la información contenida en el dataset obedece a las siguientes reglas estandarizadas y diccionarios de datos.

## Estructura General de las Columnas

| Columna | Descripción | Tipo de Dato / Valores Posibles |
|---------|-------------|--------------------------------|
| **`message_id`** | Identificador único del mensaje (ej. `msg_1a2b3c4d`). | `String (Hash)` |
| **`canonical_label`** | Etiqueta unificada definitiva. | `Categoría` (Ver sección 3) |
| **`text_raw`** | Mensaje original en bruto sin alterar, tal y como proviene de la fuente (con posibles ofuscaciones e información PII). | `String` |
| **`text`** | Mensaje procesado y normalizado tras mitigar ofuscaciones, pero conservando los enlaces y teléfonos reales (texto procesado). | `String` |
| **`text_anonymized`** | Mensaje normalizado tras aplicar adicionalmente el enmascaramiento de privacidad (PII sustituida por tokens). | `String` |
| **`reference`** | Nombre del dataset o fuente de donde proviene. | `Categoría` (Ver sección 4) |
| **`source_id`** | ID original del mensaje en su dataset de origen. | `String` o Vacío |
| **`original_label`** | La etiqueta original que tenía el mensaje en su origen. | `String` |
| **`label_mapping_rule`** | Regla aplicada para transformar la etiqueta original. | `String` |
| **`is_ai_generated`** | Indica si el mensaje ha sido generado sintéticamente por Inteligencia Artificial (0 o 1). | `Booleano (0 o 1)` |
| **`timestamp_original`** | Fecha y hora original del mensaje (si estaba disponible). | `String (ISO 8601)` o Vacío |
| **`license`** | Licencia de uso del mensaje/dataset original. | `String` |
| **`language`** | Idioma principal detectado. | `Categoría` (Ver sección 5) |
| **`language_confidence`** | Nivel de confianza de la detección del idioma. | `Float (0.0 - 1.0)` o `"LLM"` |
| **`anonymization_status`** | Estado de anonimización del texto. | `Categoría` (Ver sección 2) |
| **`llm_model`** | El modelo usado para enriquecer la fila. | `String` (ej. `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`) |
| **`prompt_version`** | La versión del prompt que se utilizó en el LLM. | `String` |
| **`llm_annotation_date`** | Fecha en la que el LLM procesó esta fila. | `String (ISO 8601)` |
| **`theme`** | Categoría o intención principal del mensaje. | `Categoría` (Ver sección 6) |
| **`urgency_level`** | Nivel de urgencia que transmite el mensaje. | `Categoría` (Ver sección 7) |
| **`whois_domain`** | Dominio extraído del enlace (si existe) tras la resolución de Whois. | `String` (o `"ERROR"`) o Vacío |
| **`whois_tld`** | Dominio de nivel superior (Top Level Domain) extraído del dominio. | `String` (o `"ERROR"`) o Vacío |
| **`whois_age_days`** | Edad en días del dominio extraída mediante Whois. | `Integer` o Vacío |
| **`whois_hidden`** | Indica si los detalles de Whois del dominio están ocultos/privados (1 o 0). | `Booleano (0 o 1)` o Vacío |
| **`whois_country`** | País de registro del dominio obtenido vía Whois. | `String` o Vacío |

---

## Diccionarios de Valores y Categorías

### 1. Variables Booleanas
Varias columnas utilizan formato booleano estandarizado con valores enteros (0 y 1) para facilitar su procesamiento en Machine Learning:
- **`0`**: Falso (No aplica)
- **`1`**: Verdadero (Sí aplica)

*Aplica a las columnas: `whois_hidden`, `is_ai_generated`.*

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
- `mimics_3500`
- `mishra_extended`
- `mishra_soni_2022`
- `nus_sms`
- `smishing_4c`
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
Clasifica el propósito o temática central del mensaje. El LLM encasilla obligatoriamente el mensaje en una sola de estas 14 categorías (taxonomía cerrada) para todo el dataset:
- **`personal`**: Conversación casual, saludos, preguntas entre personas (ej. "What's up?").
- **`family_emergency`**: Suplantación de familiares reclamando roturas de teléfono o solicitando dinero urgente (ej. "Hola mamá, este es mi nuevo número").
- **`banking`**: Bancos, transferencias, tarjetas de crédito, criptomonedas.
- **`delivery`**: Paquetería, correos, aduanas, seguimiento de envíos.
- **`account_security`**: Bloqueo de cuentas, inicios de sesión sospechosos, verificación de identidad.
- **`tech_support`**: Soporte técnico fraudulento, alertas de virus o infecciones falsas de dispositivos.
- **`promotion`**: Premios, loterías, descuentos, ofertas comerciales.
- **`survey`**: Peticiones de participación en encuestas, cuestionarios o estudios de mercado.
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
| `message_id` | `msg_1b83648c69bd` | Identificador hexadecimal único (UUID v4 aleatorio) asignado a la muestra. |
| `canonical_label` | `smishing` | Categoría unificada (ataque de phishing vía SMS). |
| `text_raw` | `Banco Santander: Su tarjeta ha sido limitada. Para reactivarla visite http://bit.ly/sant-phish` | Texto original en bruto recibido directamente de la fuente. |
| `text` | `Banco Santander: Su tarjeta ha sido limitada. Para reactivarla visite http://bit.ly/sant-phish` | Texto normalizado (en este ejemplo no contenía marcas de ofuscación). |
| `text_anonymized` | `Banco Santander: Su tarjeta ha sido limitada. Para reactivarla visite <URL>` | Texto tras enmascarar la URL maliciosa por la meta-marca unificada `<URL>`. |
| `reference` | `mimics_3500` | Proviene del repositorio web de origen. |
| `source_id` | `102394` | ID de la sumisión original en la base de datos de origen. |
| `original_label` | `7class=...;13class=...;dataset=...` | Etiqueta interna compleja que traía el dataset original. |
| `label_mapping_rule` | `implicit_smishing` | Regla determinista aplicada en el módulo de Python (mimics es 100% smishing). |
| `is_ai_generated` | `0` | Indica que el mensaje no fue generado por una Inteligencia Artificial (es orgánico). |
| `timestamp_original` | | No constaba la hora original en el dataset de origen. |
| `license` | `Academic Use Only` | Condición de uso estipulada por la fuente original. |
| `language` | `es` | Idioma español ISO 639-1 (Detectado por el motor FastText). |
| `language_confidence` | `1.00` | 100% de confianza del modelo detector de lenguaje. |
| `anonymization_status` | `anonymized` | Nuestro script aplicó Regex en fase 1 para limpiar PII. |
| `llm_model` | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | Modelo local de Hugging Face y vLLM utilizado para enriquecer este mensaje. |
| `prompt_version` | `v1.0-local` | Versión interna del script que construye los prompts de anotación. |
| `llm_annotation_date` | `2026-05-28` | Fecha en la que el LLM nos devolvió el resultado. |
| **`theme`** | `banking` | Encaja exactamente en la taxonomía bancaria. |
| **`urgency_level`** | `high` | Nivel máximo de urgencia por amenaza explícita de limitación de tarjeta. |
| **`whois_domain`** | `bit.ly` | Dominio extraído de la URL. |
| **`whois_tld`** | `ly` | TLD del dominio. |
| **`whois_age_days`** | `4500` | Edad del dominio en días. |
| **`whois_hidden`** | `0` | El Whois no está oculto. |
| **`whois_country`** | `US` | País del registrador. |

---

## Anexo: Prompt Maestro (LLM)

Para garantizar la reproducibilidad y la transparencia en la investigación (trazabilidad del campo `prompt_version`), a continuación se expone el prompt exacto de sistema (versión **`v1.0-local`**) utilizado para extraer las características `theme`, `urgency_level` y `language` mediante Zero-Shot Learning y Guided JSON Decoding (optimizando latencia eliminando explicaciones co-think):

```text
You are an expert cybersecurity analyst annotating SMS messages for a machine learning dataset.

Analyze the semantic meaning and return a JSON object with EXACTLY these keys:
- "theme": strictly one of: personal, family_emergency, banking, delivery, account_security, tech_support, promotion, survey, government, job_offer, dating_adult, subscription, service_alert, unknown
- "urgency_level": strictly one of: none, low, medium, high
- "language": 2-letter ISO 639-1 code (e.g., "en", "es", "fr")

Theme definitions:
- personal: Casual conversation, greetings, family matters.
- family_emergency: Fake relatives asking for money ("Hi mom, new number").
- banking: Banks, transfers, cards, crypto, financial alerts.
- delivery: Packages, post office, customs, shipping.
- account_security: Account locks, suspicious logins, OTP codes.
- tech_support: Fake tech support, virus alerts.
- promotion: Prizes, lotteries, discounts, aggressive offers.
- survey: Requests to complete surveys or feedback.
- government: Traffic fines, taxes, public administration.
- job_offer: Recruitment, work from home, easy money scams.
- dating_adult: Dating, sexual content, adult contacts.
- subscription: Premium SMS, horoscopes, ringtones, paid services.
- service_alert: Medical appointments, utility bills, carrier reminders.
- unknown: Unreadable or doesn't fit elsewhere.

Urgency definitions:
- none: No urgency. Normal chat or passive info.
- low: Informative, future action, no time pressure.
- medium: Attention soon ("reply when you can", "arrives tomorrow").
- high: Immediate action. Psychological pressure, threats, short deadlines.

Return ONLY the raw JSON object. No markdown, no explanations.
```

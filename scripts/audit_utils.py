"""Auditoria profunda de los utils (cleaner + anonymizer) y su interaccion con build_metasms."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src/dataset')

from utils.cleaner import DatasetCleaner
from utils.anonymizer import SMSAnonymizer
from build_metasms import (
    clean_for_llm, build_dedupe_key, detect_pre_anonymized,
    clean_str
)

cleaner = DatasetCleaner()
anon = SMSAnonymizer(use_whois=False, use_privacy_filter=False)

print("=" * 70)
print("TEST 1: OBF tags se añaden al texto_anonymized → afectan al dedupe key")
print("=" * 70)

msg_obf   = "C\u200bo\u200br\u200br\u200be\u200bo\u200bs: paquete retenido. Ver http://evil.co/abc"
msg_clean = "Correos: paquete retenido. Ver http://evil.co/abc"

for label, msg in [("con ZWSP", msg_obf), ("sin ZWSP", msg_clean)]:
    llm_text = clean_for_llm(msg, cleaner, anon)
    key = build_dedupe_key(llm_text)
    print(f"  [{label}]")
    print(f"    original  : {repr(msg[:60])}")
    print(f"    llm_text  : {repr(llm_text[:80])}")
    print(f"    dedupe_key: {key[:20]}...")
print()
print("  -> Mismos mensajes dan DISTINTO dedupe_key? :", 
      build_dedupe_key(clean_for_llm(msg_obf, cleaner, anon)) !=
      build_dedupe_key(clean_for_llm(msg_clean, cleaner, anon)))

print()
print("=" * 70)
print("TEST 2: anonymizer str.replace doble-sustitución")
print("=" * 70)

# Un mensaje con una URL corta que también puede matchear con el patron de shortcode
msg_url = "Tu codigo es 88039 y tu enlace es http://phish.co/abc"
result = anon.process_message(msg_url)
print(f"  original:   {msg_url}")
print(f"  anonymized: {result['anonymized']}")
print(f"  entities:   {result['entities']}")

# Caso problemático: URL ya reemplazada por <URL> y luego el siguiente patron vuelve a matchear
msg2 = "Visita http://t.co/abc123 o llama al 600123456"
result2 = anon.process_message(msg2)
print(f"\n  original:   {msg2}")
print(f"  anonymized: {result2['anonymized']}")

# Caso: la misma URL aparece dos veces - str.replace sustituye TODAS las ocurrencias
msg3 = "Link: http://evil.co/xyz - Reenviar: http://evil.co/xyz"
result3 = anon.process_message(msg3)
print(f"\n  original:   {msg3}")
print(f"  anonymized: {result3['anonymized']}")
print(f"  entities count: {len(result3['entities'])} (esperado 2, o 1 si str.replace borra ambas de golpe)")

print()
print("=" * 70)
print("TEST 3: detect_pre_anonymized vs cleaner: orden de operaciones")
print("=" * 70)

# Agarwal tiene <NAMED_ENTITY>, <DATE_TIME>, <URL> etc - distintos tokens
msg_agarwal = "Desculpe, nao conseguimos encontra-lo hoje (<DATE_TIME>). <NAMED_ENTITY> sua entrega aqui. <URL>"
pre_anon = detect_pre_anonymized(msg_agarwal)
cleaned = cleaner.clean(msg_agarwal)["cleaned"]
llm = clean_for_llm(msg_agarwal, cleaner, anon)
print(f"  original text: {msg_agarwal[:80]}")
print(f"  detect_pre_anonymized: {pre_anon}")
print(f"  after cleaner: {cleaned[:80]}")
print(f"  final llm_text: {llm[:80]}")
# Notar que <DATE_TIME> y <NAMED_ENTITY> NO estan en los patrones del cleaner
# -> quedan tal cual en el texto, lo cual es correcto
print()

# Problema: el cleaner normaliza <PHONE_NUMBER> -> <PHONE> pero el anonymizer
# no matchea <PHONE> (ya es un token), así que queda <PHONE> en el texto_anonymized
msg_phone_token = "Su referencia es <PHONE_NUMBER> y el codigo <PHONE_SHORTCODE>"
pre_anon2 = detect_pre_anonymized(msg_phone_token)
llm2 = clean_for_llm(msg_phone_token, cleaner, anon)
print(f"  token msg: {msg_phone_token}")
print(f"  detect_pre_anonymized: {pre_anon2}")
print(f"  llm_text: {llm2}")

print()
print("=" * 70)
print("TEST 4: label_mapping_rule en build_master_row cuando original==canonical")
print("=" * 70)

from build_metasms import build_master_row

# Simular un row de mishra donde label='ham', canonical='ham'
raw_ham = {
    "text": "Hello, how are you doing today?",
    "canonical_label": "ham",
    "original_label": "ham",
    "label_mapping_rule": "ham->ham",
    "source": "mishra_soni_2022",
    "source_id": None,
    "source_url": "https://example.com",
    "license": "Research Use",
    "language_hint": "en",
}
result_ham, _ = build_master_row(raw_ham, cleaner, anon)
print(f"  original_label='ham', canonical='ham'")
print(f"  label_mapping_rule in output: {repr(result_ham['label_mapping_rule'])}")
print(f"  -> El loader puso 'ham->ham' pero build_master_row lo ignora y pone '' (vacío)")

print()
print("=" * 70)
print("TEST 5: is_ai_generated field en loaders que NO lo tienen")
print("=" * 70)

raw_no_ai = {
    "text": "Win a prize now!",
    "canonical_label": "spam",
    "original_label": "spam",
    "source": "kaggle_spam_ham",
    "source_id": None,
    "source_url": None,
    "license": "Unknown",
    "language_hint": None,
}
result_no_ai, _ = build_master_row(raw_no_ai, cleaner, anon)
print(f"  is_ai_generated value type: {type(result_no_ai['is_ai_generated'])}")
print(f"  is_ai_generated value: {result_no_ai['is_ai_generated']}")
print(f"  bool(0) is falsy -> filter works correctly: {not bool(result_no_ai['is_ai_generated'])}")

print()
print("=" * 70)
print("TEST 6: Agarwal <NAMED_ENTITY> tokens - no mapeados por cleaner")
print("=" * 70)
# El cleaner normaliza: <USER> -> <PERSON>, <PHONE_NUMBER> -> <PHONE>
# Pero <NAMED_ENTITY>, <DATE_TIME>, <URL> de Agarwal:
#   - <URL> SI se mapea (cleaner linea 90)
#   - <NAMED_ENTITY> NO se mapea -> queda tal cual
#   - <DATE_TIME> NO se mapea -> queda tal cual
msg_ag = "Visita <NAMED_ENTITY> hoy <DATE_TIME> en <URL> o llama <PHONE_NUMBER>"
cleaned_ag = cleaner.clean(msg_ag)["cleaned"]
llm_ag = clean_for_llm(msg_ag, cleaner, anon)
print(f"  input:   {msg_ag}")
print(f"  cleaned: {cleaned_ag}")
print(f"  llm:     {llm_ag}")
print()
print("Tokens no normalizados de Agarwal que quedan en texto_anonymized:")
import re
tokens_in_text = re.findall(r'<[A-Z0-9_]+>', llm_ag)
print(f"  {tokens_in_text}")

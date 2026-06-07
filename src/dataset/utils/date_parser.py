import re
from dateutil.parser import parse, UnknownTimezoneWarning
import warnings

# Suppress warnings about unparseable timezones (e.g. IST, EST) 
# since we fallback gracefully and don't want to pollute logs.
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
from typing import Optional

def normalize_timestamp(ts: Optional[str]) -> str:
    """
    Normalizes a timestamp string. 
    Attempts to extract a full ISO-8601 datetime.
    If it's relative (e.g., 'hoy 16:45', 'Yesterday 11:29 AM'), extracts just the HH:MM:SS.
    Returns an empty string if it cannot be parsed.
    """
    if not ts or not str(ts).strip() or str(ts).lower() in ['nan', 'null', 'none']:
        return ""
    
    ts = str(ts).strip()
    
    # 1. Try to parse as full datetime
    try:
        dt = parse(ts, fuzzy=False)
        # Verify it has some date indicator (year, month name, or slash/dash format)
        # to avoid parsing "10:08 AM" as today's date.
        has_date = re.search(r'[a-zA-Z]{3,}|\d{4}|\d{1,2}[/-]\d{1,2}', ts)
        if has_date:
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
    except Exception:
        pass
        
    # 2. Try to extract just the time
    # Matches: 16:45, 10:08 AM, 4:10 PM, 08:16:00
    time_match = re.search(r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?)\b', ts)
    if time_match:
        time_str = time_match.group(1)
        try:
            dt_time = parse(time_str)
            return dt_time.strftime('%H:%M:%S')
        except Exception:
            return time_str
            
    return ""

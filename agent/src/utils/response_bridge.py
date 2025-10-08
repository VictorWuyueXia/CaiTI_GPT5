import re
import json
from src.response_analyzer import classify_dimension_and_score
from src.utils.log_util import get_logger
logger = get_logger("ResponseBridge")

def _normalize_dim_score(dim: str, score: int):
    """
    Normalize dimension label to start with DLA_ when appropriate and
    validate score range.
    """
    if re.match(r"^\d+_[A-Za-z_]+$", dim):
        dim = "DLA_" + dim
    elif not dim.startswith("DLA_"):
        # invalid dim prefix
        return None
    if not isinstance(score, int) or score < 0 or score > 2:
        return None
    return dim, score

def _parse_dim_score_from_text(text: str):
    """
    Parse '[dim][sep][score]' from a free-form text line.
    Accept separators: comma, colon, hyphen, or whitespace.
    """
    # direct, lenient separators
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*[,:\-\s]\s*([0-2])\s*$", text)
    if not m:
        # search anywhere as a fallback
        m = re.search(r"((?:DLA_)?\d+_[A-Za-z_]+)\s*[,:\-\s]\s*([0-2])", text)
    if m:
        dim = m.group(1).strip()
        score = int(m.group(2))
        norm = _normalize_dim_score(dim, score)
        if norm:
            logger.debug(f"Parsed dimension-score (text): {norm[0]}, {norm[1]}")
            return norm
    return None

def _parse_from_json_like(raw: str):
    """
    If the model returns JSON-like content, try to extract:
    - {'res': '3_talk, 1'}
    - {'dimension': '3_talk', 'score': 1}
    """
    s = str(raw).strip()
    if s.startswith("{") and s.endswith("}"):
        data = json.loads(s)
        logger.debug(f"Parsed JSON classification: {data}")
        # normalize keys
        kl = {str(k).lower(): v for k, v in data.items()}
        # case 1: 'res' key contains a string like '3_talk, 1'
        if "res" in kl:
            val = str(kl["res"]).strip()
            got = _parse_dim_score_from_text(val)
            if got:
                return got
        # case 2: separate 'dimension' and 'score' keys
        if "dimension" in kl and "score" in kl:
            dim = str(kl["dimension"]).strip()
            sc = int(kl["score"])
            norm = _normalize_dim_score(dim, sc)
            if norm:
                logger.debug(f"Parsed dimension-score (json): {norm[0]}, {norm[1]}")
                return norm
        # otherwise, scan stringified JSON for dim/score
        got = _parse_dim_score_from_text(s)
        if got:
            return got
    return None

def get_openai_resp(user_input):
    lower = [t.lower() for t in user_input.replace(".", " ").replace(",", " ").replace("?", " ").split()[:5]]
    if "stop" in lower: return "DLA", "Stop"
    if "yes" in lower:  return "DLA", "Yes"
    if "no"  in lower:  return "DLA", "No"
    try:
        raw = classify_dimension_and_score(user_input)
        first = str(raw).strip().splitlines()[0].strip()
        logger.debug(f"OpenAI raw: {raw}")
        logger.debug(f"First line parsed: {first}")
    except Exception as e:
        logger.debug(f"classify_dimension_and_score exception: {e}")
        return "NA", 99

    # handle general tokens like Yes/No/Stop/Question/Maybe
    m = re.match(r"^\s*(Yes|No|Stop|Question|Maybe)\s*,?\s*(\d+)?\s*$", first, flags=re.IGNORECASE)
    if m:
        token = m.group(1).strip().lower()
        if token == "maybe": return "DLA", "Question"
        if token in ("yes","no","stop","question"):
            return "DLA", token.capitalize()

    # JSON-like payloads
    got = _parse_from_json_like(first)
    if not got:
        got = _parse_from_json_like(str(raw))
    if got:
        return got

    # Plain text dim, score
    got = _parse_dim_score_from_text(first)
    if got:
        return got

    # 'Other, N' is treated as NA,99
    m = re.match(r"^\s*(Other)\s*,\s*(\d+)\s*$", first, flags=re.IGNORECASE)
    if m:
        return "NA", 99

    logger.debug("Failed to parse classification, fallback to NA,99")
    return "NA", 99
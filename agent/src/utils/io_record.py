import os
import logging
import time
import pandas as pd
from pandas.errors import EmptyDataError
from src.utils.config_loader import RECORD_CSV

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("IORecord")

HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]

def _read():
    last_exc = None
    for _ in range(5):
        try:
            time.sleep(0.03)
            return pd.read_csv(RECORD_CSV, dtype={"Question": str, "Question_Lock": "int64", "Resp": str, "Resp_Lock": "int64"})
        except (EmptyDataError, FileNotFoundError, OSError) as e:
            last_exc = e
            time.sleep(0.05)
    raise last_exc

def _write(df):
    time.sleep(0.03)
    folder = os.path.dirname(RECORD_CSV)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    tmp_path = RECORD_CSV + ".tmp"
    df.to_csv(tmp_path, columns=HEADER, index=False)
    os.replace(tmp_path, RECORD_CSV)
    time.sleep(0.03)

def log_question(text: str):
    while True:
        time.sleep(0.1)
        data = _read()
        if data.loc[0, "Question_Lock"] == 0:
            data.loc[0, "Question"] = text
            data.loc[0, "Question_Lock"] = 1
            _write(data)
            logger.info(f"Prompted question: {text}")
            break

def get_answer():
    while True:
        time.sleep(0.1)
        data = _read()
        if data.loc[0, "Resp_Lock"] == 0:
            user_input = data.loc[0, "Resp"]
            data.loc[0, "Resp_Lock"] = 1
            _write(data)
            break
    user_input = str(user_input)
    user_input = user_input.replace(", and", ".").replace("but", ".")
    user_input = user_input.split(".")
    DLA_result, segments = [], []
    for seg in user_input:
        if not seg:
            continue
        if seg[0] == " ":
            seg = seg[1:]
        segments.append(seg)
    return DLA_result, segments

def get_resp_log():
    while True:
        time.sleep(0.1)
        data = _read()
        if data.loc[0, "Resp_Lock"] == 0:
            follow = data.loc[0, "Resp"]
            data.loc[0, "Resp_Lock"] = 1
            _write(data)
            logger.info(f"Received response: {follow}")
            break
    return follow

def init_record():
    try:
        data = _read()
    except FileNotFoundError:
        data = pd.DataFrame([["", 0, "", 1]], columns=HEADER)
        _write(data)
    time.sleep(0.03)
    data.loc[0, 'Question_Lock'] = 0
    data.loc[0, 'Resp_Lock'] = 1
    _write(data)



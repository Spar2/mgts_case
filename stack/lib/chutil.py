"""Общая CH-утилита backend/ingester: вставка JSONEachRow по HTTP.

ClickHouse не парсит float в DateTime64 из JSONEachRow — время идёт строкой
'YYYY-MM-DD HH:MM:SS.ffffff' (UTC). JSON не умеет nonfinite — null по контракту.
"""

import json
import math
import urllib.parse
import urllib.request
from datetime import UTC, datetime


def epoch_to_ch(ts):
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def sanitize_temps(temps):
    return [t if math.isfinite(t) else None for t in temps]


def ch_insert(ch_url, table, rows, timeout=30):
    if not rows:
        return
    body = "\n".join(json.dumps(r) for r in rows)
    q = f"INSERT INTO haklolarp.{table} FORMAT JSONEachRow"
    req = urllib.request.Request(
        ch_url + "/?query=" + urllib.parse.quote(q),
        data=body.encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def ch_query(ch_url, query, parameters=None, timeout=30):
    params={"query":query+" FORMAT JSONEachRow",**{"param_"+k:str(v) for k,v in (parameters or {}).items()}}
    with urllib.request.urlopen(ch_url+"/?"+urllib.parse.urlencode(params),timeout=timeout) as response:
        return [json.loads(line) for line in response.read().decode().splitlines() if line]


def ch_insert_native(ch_url, rows, timeout=30):
    """RowBinary avoids decimal JSON parsing drift in native Float64 coordinates."""
    import struct
    def varint(value):
        result=bytearray()
        while value>=128:
            result.append((value&127)|128);value>>=7
        result.append(value);return bytes(result)
    def string(value):
        encoded=value.encode('utf-8');return varint(len(encoded))+encoded
    columns=('replay_id','source_sha256','dts_id','channel_id','ts_raw','ts','time_kind','elapsed_s','row_idx','segment_id','x_m','temp_c','invalid_reason','payload_sha256')
    body=bytearray()
    for r in rows:
        for key in columns[:5]:body.extend(string(r[key]))
        stamp=r['ts'];body.append(1 if stamp is None else 0)
        if stamp is not None:
            epoch=datetime.fromisoformat(stamp).replace(tzinfo=UTC).timestamp();body.extend(struct.pack('<q',round(epoch*1000000)))
        body.extend(string(r['time_kind']));body.extend(struct.pack('<dII',r['elapsed_s'],r['row_idx'],r['segment_id']))
        body.extend(varint(len(r['x_m'])))
        for value in r['x_m']:body.extend(struct.pack('<d',value))
        body.extend(varint(len(r['temp_c'])))
        for value in r['temp_c']:
            body.append(1 if value is None else 0)
            if value is not None:body.extend(struct.pack('<d',value))
        body.extend(varint(len(r['invalid_reason'])));body.extend(bytes(r['invalid_reason']));body.extend(string(r['payload_sha256']))
    query='INSERT INTO haklolarp.profiles_v2 ('+','.join(columns)+') FORMAT RowBinary'
    request=urllib.request.Request(ch_url+'/?query='+urllib.parse.quote(query),data=bytes(body),method='POST')
    with urllib.request.urlopen(request,timeout=timeout) as response:response.read()

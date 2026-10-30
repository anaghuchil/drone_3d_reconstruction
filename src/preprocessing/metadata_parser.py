import os
import re
import pandas as pd


def _srt_time_to_seconds(time_str: str) -> float:
    time_str = time_str.strip().replace(".", ",")
    parts = time_str.split(":")
    if len(parts) != 3:
        return 0.0
    h, m, rest = parts
    if "," in rest:
        s, ms = rest.split(",", 1)
    else:
        s, ms = rest, "0"
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms[:3].ljust(3, "0")) / 1000.0


class MetadataParser:
    @staticmethod
    def parse_dji_srt(srt_path: str) -> pd.DataFrame:
        if not srt_path or not os.path.exists(srt_path):
            return pd.DataFrame()

        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        content = content.replace("\r\n", "\n").replace("\r", "\n")
        blocks = [b for b in content.strip().split("\n\n") if b.strip()]

        records = []
        for block in blocks:
            lines = block.split("\n")
            if len(lines) < 2:
                continue

            time_line = next((l for l in lines if "-->" in l), None)
            if not time_line:
                continue

            try:
                timestamp_sec = _srt_time_to_seconds(time_line.split("-->")[0])
            except Exception:
                continue

            lat_m = re.search(r"\[?latitude\s*:\s*([-\d\.]+)\]?", block, re.I)
            lon_m = re.search(r"\[?longitude\s*:\s*([-\d\.]+)\]?", block, re.I)
            alt_m = re.search(r"\[?(?:rel_alt|abs_alt|altitude)\s*:\s*([-\d\.]+)\]?", block, re.I)

            if not (lat_m and lon_m):
                gps_tuple = re.search(
                    r"GPS\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)",
                    block,
                    re.I,
                )
                if gps_tuple:
                    records.append({
                        "timestamp_sec": timestamp_sec,
                        "latitude": float(gps_tuple.group(2)),
                        "longitude": float(gps_tuple.group(1)),
                        "altitude": float(gps_tuple.group(3)),
                    })
                continue

            records.append({
                "timestamp_sec": timestamp_sec,
                "latitude": float(lat_m.group(1)),
                "longitude": float(lon_m.group(1)),
                "altitude": float(alt_m.group(1)) if alt_m else 0.0,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            print(f"[+] Parsed {len(df)} GPS telemetry records from SRT: {srt_path}")
        return df

    @staticmethod
    def parse_flight_csv(csv_path: str) -> pd.DataFrame:
        if not csv_path or not os.path.exists(csv_path):
            return pd.DataFrame()

        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]

        required = ["latitude", "longitude", "altitude"]
        if not all(col in df.columns for col in required):
            print(f"[!] CSV {csv_path} missing required columns {required}.")
            return pd.DataFrame()

        if "timestamp_sec" not in df.columns:
            if "timestamp" in df.columns:
                df["timestamp_sec"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
            else:
                df["timestamp_sec"] = df.index.astype(float)

        print(f"[+] Parsed {len(df)} flight log records from CSV: {csv_path}")
        return df[["timestamp_sec", "latitude", "longitude", "altitude"]]

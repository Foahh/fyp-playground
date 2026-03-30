"""Extract STM32N6 NPU metrics from stm32ai-modelzoo object_detection README files into CSV."""

from __future__ import annotations

import csv
import re
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlparse

import markdown
from bs4 import BeautifulSoup

from .constants import (
    BASE_DIR,
    CSV_COLUMNS_NO_POWER,
    METRIC_PARSED_CSV_PATH,
    MODEL_REGISTRY,
    MODELZOO_DIR,
    ensure_dirs,
)


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", h.replace("\n", " ").strip().lower())


def _norm_dataset_readme(s: str) -> str:
    s = (s or "").strip()
    if s == "COCO":
        return "COCO-80"
    if s == "COCO-80-classes":
        return "COCO-80"
    return s


def _norm_model_basename(name: str) -> str:
    """Match registry filenames to README links (e.g. drop -st before .tflite)."""
    n = name.lower().strip()
    for ext in (".tflite", ".onnx", ".keras"):
        suf = "-st" + ext
        if n.endswith(suf):
            n = n[:-len(suf)] + ext
            break
    return n


def _href_basename(href: str) -> str:
    if not href:
        return ""
    p = href.split("?")[0].strip()
    if p.startswith("http://") or p.startswith("https://"):
        path = urlparse(p).path
        return unquote(Path(path).name)
    return unquote(Path(p).name)


def _href_path_key(href: str) -> str:
    """Stable key for README links: relative path for zoo models, basename for http(s)."""
    if not href:
        return ""
    p = href.split("?")[0].strip()
    if p.startswith("http://") or p.startswith("https://"):
        return _norm_model_basename(_href_basename(p))
    return unquote(p).replace("\\", "/").lower().lstrip("./")


def _registry_model_key(reg: dict) -> str:
    """Key aligned with _href_path_key for files under object_detection/<family>/."""
    model_path = (BASE_DIR / reg["model"]).resolve()
    fam_dir = (MODELZOO_DIR / reg["family"]).resolve()
    try:
        rel = model_path.relative_to(fam_dir)
        return str(rel).replace("\\", "/").lower()
    except ValueError:
        return _norm_model_basename(model_path.name)


def _first_link_href(cell) -> str:
    a = cell.find("a")
    if not a:
        return ""
    return unescape(a.get("href") or "").strip()


def _parse_md_tables(md_text: str) -> BeautifulSoup:
    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    html = md.convert(md_text)
    return BeautifulSoup(html, "html.parser")


def _table_to_rows(table) -> tuple[list[str], list[list[str]]]:
    """Return (header_texts, data_rows) from a <table>."""
    rows = table.find_all("tr")
    if not rows:
        return [], []
    header_cells = rows[0].find_all(["th", "td"])
    headers = [_norm_header(c.get_text()) for c in header_cells]
    if not headers:
        return [], []
    body: list[list[str]] = []
    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        body.append([c.get_text(strip=True) for c in cells])
    return headers, body


def _header_indices(headers: list[str]) -> dict[str, int]:
    return {h: i for i, h in enumerate(headers)}


def _find_hyperparameter_col(headers: list[str]) -> int | None:
    for i, h in enumerate(headers):
        if "hyperparameter" in h:
            return i
    return None


def _is_npu_memory_table(headers: list[str]) -> bool:
    joined = " ".join(headers)
    return "internal ram" in joined and "series" in joined and "weights flash" in joined


def _is_npu_inference_table(headers: list[str]) -> bool:
    joined = " ".join(headers)
    return (
        "inference time" in joined
        and "board" in joined
        and ("inf" in joined or "sec" in joined)
    )


def _find_ap_col_index(headers: list[str]) -> int | None:
    for i, h in enumerate(headers):
        hn = _norm_header(h)
        if hn in ("ap50", "ap*", "ap"):
            return i
        if hn.startswith("ap") and "map" not in hn:
            return i
    return None


def _parse_resolution_cell(text: str) -> int | None:
    m = re.match(r"^(\d+)x\d+x\d+", (text or "").strip())
    if m:
        return int(m.group(1))
    m2 = re.match(r"^(\d+)x\d+", (text or "").strip())
    if m2:
        return int(m2.group(1))
    m3 = re.match(r"^3x(\d+)x(\d+)", (text or "").strip())
    if m3:
        return int(m3.group(2))
    return None


def _fmt_norm_cell(fmt: str) -> str:
    s = (fmt or "").strip().lower()
    if s == "w4w8":
        return "w4a8"
    return s


def _plain_ap_key(family: str, fmt_cell: str, res_cell: str) -> str:
    r = _parse_resolution_cell(res_cell)
    if r is None:
        return ""
    return f"{family}::{_fmt_norm_cell(fmt_cell)}::{r}"


def _registry_plain_ap_key(reg: dict) -> str:
    return f"{reg['family']}::{_fmt_norm_cell(reg['fmt'])}::{reg['resolution']}"


def _extract_family_metrics(readme_path: Path) -> dict[str, dict]:
    """
    Build map: norm_model_basename -> merged metrics dict with keys:
    link_key, internal_ram_kib, external_ram_kib, weights_flash_kib,
    inference_time_ms, inf_per_sec, stedgeai_version, ap_50 (optional),
    dataset (normalized), format, hyperparameter (optional), resolution (from table).
    """
    if not readme_path.is_file():
        return {}

    family = readme_path.parent.name
    md_text = readme_path.read_text(encoding="utf-8")
    soup = _parse_md_tables(md_text)

    mem_by_key: dict[str, dict] = {}
    inf_by_key: dict[str, dict] = {}
    ap_entries: list[dict] = []

    for table in soup.find_all("table"):
        headers, body = _table_to_rows(table)
        if not headers or not body:
            continue

        hi = _header_indices(headers)
        hyp_idx = _find_hyperparameter_col(headers)

        if _is_npu_memory_table(headers):
            si = hi.get("series")
            if si is None:
                continue
            ir = hi.get("internal ram (kib)") or hi.get("internal ram")
            er = hi.get("external ram (kib)") or hi.get("external ram")
            wf = hi.get("weights flash (kib)") or hi.get("weights flash")
            ds = hi.get("dataset")
            fmt = hi.get("format")
            res = hi.get("resolution")
            stv = hi.get("stedgeai core version")
            if ir is None or wf is None:
                continue
            for row_idx, row_cells in enumerate(body):
                if si >= len(row_cells):
                    continue
                if row_cells[si].strip().upper() != "STM32N6":
                    continue
                tr = table.find_all("tr")[row_idx + 1]
                tds = tr.find_all(["td", "th"])
                if not tds:
                    continue
                href = _first_link_href(tds[0])
                key = _href_path_key(href)
                if not key:
                    continue
                rec: dict = {
                    "link_key": key,
                    "internal_ram_kib": row_cells[ir].strip() if ir < len(row_cells) else "",
                    "external_ram_kib": row_cells[er].strip()
                    if er is not None and er < len(row_cells)
                    else "0",
                    "weights_flash_kib": row_cells[wf].strip()
                    if wf < len(row_cells)
                    else "",
                }
                if ds is not None and ds < len(row_cells):
                    rec["dataset"] = _norm_dataset_readme(row_cells[ds])
                if fmt is not None and fmt < len(row_cells):
                    rec["format"] = row_cells[fmt].strip()
                if res is not None and res < len(row_cells):
                    r = _parse_resolution_cell(row_cells[res])
                    if r is not None:
                        rec["resolution_table"] = r
                if hyp_idx is not None and hyp_idx < len(row_cells):
                    rec["hyperparameter"] = row_cells[hyp_idx].strip()
                if stv is not None and stv < len(row_cells):
                    rec["stedgeai_version"] = row_cells[stv].strip()
                mem_by_key[key] = rec

        elif _is_npu_inference_table(headers):
            bi = hi.get("board")
            it = hi.get("inference time (ms)")
            ips = hi.get("inf / sec")
            ds = hi.get("dataset")
            fmt = hi.get("format")
            res = hi.get("resolution")
            stv = hi.get("stedgeai core version")
            if bi is None or it is None:
                continue
            for row_idx, row_cells in enumerate(body):
                if bi >= len(row_cells):
                    continue
                if row_cells[bi].strip() != "STM32N6570-DK":
                    continue
                tr = table.find_all("tr")[row_idx + 1]
                tds = tr.find_all(["td", "th"])
                if not tds:
                    continue
                href = _first_link_href(tds[0])
                key = _href_path_key(href)
                if not key:
                    continue
                rec = {
                    "link_key": key,
                    "inference_time_ms": row_cells[it].strip() if it < len(row_cells) else "",
                    "inf_per_sec": row_cells[ips].strip()
                    if ips is not None and ips < len(row_cells)
                    else "",
                }
                if ds is not None and ds < len(row_cells):
                    rec["dataset"] = _norm_dataset_readme(row_cells[ds])
                if fmt is not None and fmt < len(row_cells):
                    rec["format"] = row_cells[fmt].strip()
                if res is not None and res < len(row_cells):
                    r = _parse_resolution_cell(row_cells[res])
                    if r is not None:
                        rec["resolution_table"] = r
                if hyp_idx is not None and hyp_idx < len(row_cells):
                    rec["hyperparameter"] = row_cells[hyp_idx].strip()
                if stv is not None and stv < len(row_cells):
                    rec["stedgeai_version"] = row_cells[stv].strip()
                inf_by_key[key] = rec

        else:
            ap_i = _find_ap_col_index(headers)
            if ap_i is None or not headers or headers[0] != "model":
                continue
            mi = hi.get("model")
            if mi is None:
                mi = 0
            fmt_i = hi.get("format")
            res_i = hi.get("resolution")
            for row_idx, row_cells in enumerate(body):
                if mi >= len(row_cells) or ap_i >= len(row_cells):
                    continue
                tr = table.find_all("tr")[row_idx + 1]
                tds = tr.find_all(["td", "th"])
                if not tds:
                    continue
                href = _first_link_href(tds[0])
                raw_ap = row_cells[ap_i]
                m_pct = re.search(r"([\d.]+)\s*%", raw_ap)
                ap_val = m_pct.group(1) if m_pct else re.sub(r"[^\d.]", "", raw_ap)
                if href:
                    key = _href_path_key(href)
                elif fmt_i is not None and res_i is not None and fmt_i < len(
                    row_cells
                ) and res_i < len(row_cells):
                    key = _plain_ap_key(
                        family, row_cells[fmt_i], row_cells[res_i]
                    )
                else:
                    continue
                if not key:
                    continue
                ent = {"link_key": key, "ap_50": ap_val.strip()}
                if fmt_i is not None and fmt_i < len(row_cells):
                    ent["format"] = row_cells[fmt_i].strip()
                ap_entries.append(ent)

    merged: dict[str, dict] = {}
    for k, m in mem_by_key.items():
        merged[k] = {**m}
    for k, inf in inf_by_key.items():
        if k not in merged:
            merged[k] = {"link_key": k}
        merged[k].update({x: inf[x] for x in inf if x != "link_key"})

    for ent in ap_entries:
        k = ent["link_key"]
        if k not in merged:
            merged[k] = {}
        if ent.get("format") and merged[k].get("format"):
            if ent["format"].lower() != merged[k]["format"].lower():
                continue
        merged[k]["ap_50"] = ent.get("ap_50", "")

    return merged


def _fmt_match(reg: dict, row_fmt: str) -> bool:
    """Compare registry fmt to README row; README sometimes mislabels qdq_int8 rows as W4A8."""
    if not row_fmt:
        return True
    a = reg["fmt"].strip().lower()
    b = row_fmt.strip().lower()
    if a == b:
        return True
    if {a, b} <= {"w4a8", "w4w8"}:
        return True
    fn = str(reg.get("model", "")).lower()
    if a == "int8" and b == "w4a8" and (
        "qdq_int8" in fn or "_int8." in fn or fn.endswith("int8.tflite")
    ):
        return True
    return False


def _dataset_match(reg_ds: str, row_ds: str | None) -> bool:
    if not row_ds:
        return True
    return _norm_dataset_readme(row_ds) == reg_ds


def _hyper_match(reg_hp: str, row_hp: str | None) -> bool:
    if not reg_hp:
        return True
    if row_hp is None:
        # README tables without a hyperparameter column (e.g. SSD, STYOLO) still match
        # when the registry lists a tag such as actrelu; disambiguation is by model link.
        return True
    return reg_hp.strip() == row_hp.strip()


def _pick_metrics(
    reg: dict, family_metrics: dict[str, dict]
) -> dict[str, str]:
    model_path = BASE_DIR / reg["model"]
    basename = model_path.name
    lookup_key = _registry_model_key(reg)
    norm_base = _norm_model_basename(basename)

    candidates: list[tuple[str, dict]] = [
        (k, v) for k, v in family_metrics.items() if k == lookup_key
    ]
    out = {
        "stedgeai_version": "",
        "internal_ram_kib": "",
        "external_ram_kib": "",
        "weights_flash_kib": "",
        "inference_time_ms": "",
        "inf_per_sec": "",
        "ap_50": "",
    }
    if not candidates:
        for k, v in family_metrics.items():
            if k == norm_base or k.endswith("/" + norm_base) or norm_base == k:
                candidates.append((k, v))
    def _fill_from_row(row: dict, check_resolution: bool) -> bool:
        if not _dataset_match(reg["dataset"], row.get("dataset")):
            return False
        if not _fmt_match(reg, row.get("format", "")):
            return False
        if not _hyper_match(reg.get("hyperparameters", ""), row.get("hyperparameter")):
            return False
        if check_resolution:
            rt = row.get("resolution_table")
            if rt is not None and int(rt) != int(reg["resolution"]):
                return False
        for field in out:
            if row.get(field):
                out[field] = str(row[field])
        return True

    for _, row in candidates:
        if _fill_from_row(row, check_resolution=True):
            break
    else:
        for _, row in candidates:
            if _fill_from_row(row, check_resolution=False):
                break

    if not out["ap_50"]:
        plain = family_metrics.get(_registry_plain_ap_key(reg))
        if plain and plain.get("ap_50"):
            out["ap_50"] = str(plain["ap_50"])
    if not out["ap_50"]:
        row = family_metrics.get(lookup_key)
        if row and row.get("ap_50"):
            out["ap_50"] = str(row["ap_50"])

    return out


def build_metric_rows() -> list[dict[str, str]]:
    cache: dict[str, dict[str, dict]] = {}
    rows: list[dict[str, str]] = []

    for reg in MODEL_REGISTRY:
        readme_rel = reg.get("readme")
        if readme_rel is None:
            family_metrics: dict[str, dict] = {}
        else:
            readme = BASE_DIR / readme_rel
            ck = str(readme)
            if ck not in cache:
                cache[ck] = _extract_family_metrics(readme)
            family_metrics = cache[ck]

        mets = _pick_metrics(reg, family_metrics)
        rows.append(
            {
                "host_time_iso": "",
                "stedgeai_version": mets["stedgeai_version"],
                "model_family": reg["family"],
                "model_variant": reg["variant"],
                "hyperparameters": reg.get("hyperparameters", "") or "",
                "dataset": reg["dataset"],
                "format": reg["fmt"],
                "resolution": str(reg["resolution"]),
                "internal_ram_kib": mets["internal_ram_kib"],
                "external_ram_kib": mets["external_ram_kib"],
                "weights_flash_kib": mets["weights_flash_kib"],
                "inference_time_ms": mets["inference_time_ms"],
                "inf_per_sec": mets["inf_per_sec"],
                "ap_50": mets["ap_50"],
            }
        )
    return rows


# README-derived values only; registry columns may still be set when these are blank.
_PARSED_METRIC_KEYS = (
    "internal_ram_kib",
    "external_ram_kib",
    "weights_flash_kib",
    "inference_time_ms",
    "inf_per_sec",
    "ap_50",
)


def _parsed_metrics_empty(row: dict[str, str]) -> bool:
    return not any((row.get(k) or "").strip() for k in _PARSED_METRIC_KEYS)


def write_metric_parsed_csv(path: Path | None = None) -> tuple[Path, int]:
    ensure_dirs()
    out = path or METRIC_PARSED_CSV_PATH
    built = build_metric_rows()
    rows: list[dict[str, str]] = []
    for r in built:
        if _parsed_metrics_empty(r):
            label = (
                f"{r.get('model_family', '')}/{r.get('model_variant', '')}"
                f" ({r.get('dataset', '')}, {r.get('format', '')}, {r.get('resolution', '')})"
            )
            print(f"Skipped saving row (no parsed README metrics): {label}")
            continue
        rows.append(r)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS_NO_POWER)
        w.writeheader()
        w.writerows(rows)
    skipped = len(built) - len(rows)
    if skipped:
        print(f"Skipped {skipped} model(s) with empty parsed metrics (not written).")
    return out, len(rows)


def main():
    p, n = write_metric_parsed_csv()
    print(f"Wrote {p} ({n} rows)")


if __name__ == "__main__":
    main()

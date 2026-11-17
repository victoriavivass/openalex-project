"""Everything OpenAlex-related in a single module: configuration, API client, filters,
labs, counts, download and storage.

Usage (with the virtual environment active, from the project root):

    python src/download.py subfields
    python src/download.py institutions
    python src/download.py counts
    python src/download.py sample --n 500 [--random]
    python src/download.py inventory --n 500
    python src/download.py download --dry-run     # volumes only, downloads nothing
    python src/download.py download               # the full download (resumable)
    python src/download.py build                  # merge everything into one parquet
    python src/download.py ratelimit              # daily budget left on this key
    python src/download.py preview --year 2018    # small readable CSV, to show the project

Global option (before the subcommand): --max-calls 200  (cap on API calls per run)

It can also be imported from a notebook:  from src.download import OpenAlexClient, ...
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

# =============================================================================
# 1. CONFIGURATION (the only section to edit when changing the sample)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
for _d in (RAW_DIR, PROCESSED_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABS_CSV = CONFIG_DIR / "labs_candidates.csv"         # reviewed by hand (column `keep`)
FILTERS_CACHE = CONFIG_DIR / "filters_resolved.json"  # filter syntax already validated

# ---- API key -------------------------------------------------------------
# Option A (simplest): paste your key between the quotes below.
# Option B (safer): leave it empty and write OPENALEX_API_KEY=your_key in the .env file.
# Get a free key at https://openalex.org/settings/api
#
# WARNING if you use option A: the key travels with this file. Blank it out before
# sharing the code, pushing to GitHub or submitting the Colab notebook.
MY_API_KEY = ""
# --------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = MY_API_KEY or os.getenv("OPENALEX_API_KEY")   # never printed
MAILTO = os.getenv("OPENALEX_MAILTO")                   # optional
BASE_URL = "https://api.openalex.org"

CS_FIELD_ID = 17                 # Computer Science
WORK_TYPE = "article"
YEARS = list(range(2014, 2022))  # 2014-2021
SEED = 42                        # fixed seed -> reproducible sampling
NON_COMPANY_PER_YEAR = 10000     # control group size per year

# Leaky columns: these must NOT be used as predictors
LEAKAGE_COLS = [
    "cited_by_count", "counts_by_year", "fwci", "citation_normalized_percentile",
    "cited_by_percentile_year", "is_in_top_1_percent", "related_works", "updated_date",
]

# Names to SEARCH for in /institutions. Starting points:
#  - Jurowetzki, Hain, Mateos-Garcia & Stathoulopoulos, "The Privatization of AI
#    Research(-ers)" (working paper, 2021) -> Figure 7 top destinations: google,
#    microsoft, ibm, yahoo, facebook, siemens, intel, adobe systems, baidu, samsung.
#  - Jurowetzki et al., AI & Society (2025) -> companies named in the text and the
#    NeurIPS 2022 corporate affiliations in footnote 2.
# NOTE: neither paper uses a curated company list to define "industry". They infer the
# affiliation type from name terms and from GRID (the predecessor of ROR), which is what
# OpenAlex's `institutions.type:company` already does. This list is ONLY used to flag the
# large labs / Big Tech subgroup. The final decision is made by hand (column `keep`).
LAB_QUERIES = [
    # US tech giants
    "Google", "Google Research", "Google Brain", "DeepMind", "Google DeepMind",
    "Microsoft", "Microsoft Research", "Microsoft Research Asia",
    "Meta", "Meta AI", "Facebook", "Facebook AI Research",
    "Amazon", "Amazon Web Services", "Apple", "IBM", "IBM Research", "Nvidia",
    "Intel", "Adobe", "Adobe Systems", "Yahoo", "Qualcomm", "Salesforce",
    "Uber", "Netflix", "Twitter", "Oracle", "Cisco", "Xerox", "SRI International",
    # AI-first labs
    "OpenAI", "Anthropic", "Hugging Face", "Stability AI", "Cohere",
    # Chinese tech
    "Huawei", "Huawei Noah's Ark Lab", "Alibaba", "Tencent", "Baidu",
    "ByteDance", "SenseTime", "Megvii", "iFlytek", "JD.com", "DiDi",
    # Korean / Japanese
    "Samsung", "LG Electronics", "Naver", "Sony", "NEC", "Hitachi", "Fujitsu",
    "Toshiba", "Panasonic", "Mitsubishi Electric", "Toyota Research Institute",
    "Preferred Networks", "NTT",
    # European / industrial (Siemens appears in Figure 7 of the 2021 paper)
    "Siemens", "Robert Bosch", "Philips", "SAP", "Ericsson", "Nokia",
    "Nokia Bell Labs", "Thales", "Airbus", "ASML", "Yandex",
    # Other research-active firms
    "Disney Research", "Bloomberg", "eBay", "Spotify", "Snap", "LinkedIn",
    "VMware", "Siemens Healthineers",
]


# =============================================================================
# 2. API CLIENT (retries, call budget, pagination)
# =============================================================================
class OpenAlexError(RuntimeError):
    pass


class BudgetExceeded(OpenAlexError):
    pass


class OpenAlexClient:
    def __init__(self, api_key=API_KEY, mailto=MAILTO, max_calls=None):
        self.api_key = api_key
        self.mailto = mailto
        self.max_calls = max_calls   # None = no cap
        self.calls = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ensae-asl-project/1.0 (academic use)"})

    def _redact(self, text):
        """Strip the API key from any text that may be displayed."""
        text = str(text)
        return text.replace(self.api_key, "***") if self.api_key else text

    def get(self, path, params=None, max_retries=6, timeout=60):
        """GET with retries on 429/5xx. Returns the decoded JSON."""
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        url = path if path.startswith("http") else f"{BASE_URL}{path}"

        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded(f"Call budget of {self.max_calls} reached.")
            self.calls += 1
            try:
                r = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise OpenAlexError(f"Network failure: {self._redact(exc)}") from None
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt == max_retries:
                    raise OpenAlexError(f"HTTP {r.status_code} after {max_retries} attempts: {r.text[:200]}")
                wait = r.headers.get("Retry-After", "")
                time.sleep(float(wait) if wait.replace(".", "", 1).isdigit() else backoff)
                backoff = min(backoff * 2, 60)
                continue
            # Any other 4xx: almost always a malformed filter -> show the body
            raise OpenAlexError(f"HTTP {r.status_code} on {path}: {r.text[:300]}")
        raise OpenAlexError("Retries exhausted")

    def count(self, filter_str):
        """meta.count from /works with per_page=1 (barely consumes quota)."""
        return self.get("/works", {"filter": filter_str, "per_page": 1})["meta"]["count"]

    def total_works(self):
        return self.get("/works", {"per_page": 1})["meta"]["count"]

    def fetch_cursor(self, filter_str, n_target, per_page=100, select=None):
        """Download up to n_target works using cursor pagination."""
        out, cursor = [], "*"
        bar = tqdm(total=n_target, desc="download (cursor)")
        while cursor and len(out) < n_target:
            params = {"filter": filter_str, "per_page": per_page, "cursor": cursor}
            if select:
                params["select"] = ",".join(select)
            page = self.get("/works", params)
            if not page["results"]:
                break
            out.extend(page["results"])
            bar.update(len(page["results"]))
            cursor = page["meta"].get("next_cursor")
        bar.close()
        return out[:n_target]

    def fetch_sample(self, filter_str, n, seed=SEED, per_page=100, select=None):
        """Reproducible RANDOM sample (sample + seed, paginated with `page`).

        `sample` cannot be combined with a cursor. Verify on a small sample that it
        returns n distinct results before relying on it at scale.
        """
        out, page_no = [], 1
        bar = tqdm(total=n, desc=f"random sample seed={seed}")
        while len(out) < n:
            params = {"filter": filter_str, "sample": n, "seed": seed,
                      "per_page": per_page, "page": page_no}
            if select:
                params["select"] = ",".join(select)
            res = self.get("/works", params)["results"]
            if not res:
                break
            out.extend(res)
            bar.update(len(res))
            page_no += 1
        bar.close()
        return out[:n]


# =============================================================================
# 3. FILTERS (validated with a count: a malformed filter returns ALL of OpenAlex)
# =============================================================================
FIELD_CANDIDATES = [
    "primary_topic.field.id:{id}",
    "primary_topic.field.id:fields/{id}",
    "topics.field.id:{id}",
]
COMPANY_CANDIDATES = [
    "institutions.type:company",
    "authorships.institutions.type:company",
]
LAB_CANDIDATES = [                              # {ids} = I123|I456|...
    "authorships.institutions.lineage:{ids}",   # includes subsidiaries
    "institutions.id:{ids}",
]
# "no company at all" = the negation of the company filter. OpenAlex negates with "!".
NO_COMPANY_CANDIDATES = [
    "institutions.type:!company",
    "authorships.institutions.type:!company",
]


def _load_cache():
    return json.loads(FILTERS_CACHE.read_text(encoding="utf-8")) if FILTERS_CACHE.exists() else {}


def _save_cache(cache):
    FILTERS_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def resolve_field_filter(client, verbose=True):
    """Return the field-filter syntax that actually filters (cached)."""
    cache = _load_cache()
    if "field" in cache:
        return cache["field"]
    total_all = client.total_works()
    for cand in FIELD_CANDIDATES:
        f = cand.format(id=CS_FIELD_ID)
        try:
            c = client.count(f"{f},type:{WORK_TYPE},publication_year:2018")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [x] {f:40s} ERROR {str(e)[:70]}")
            continue
        ok = 0 < c < total_all * 0.5
        if verbose:
            print(f"  [{'ok' if ok else '??'}] {f:40s} {c:,} CS articles in 2018")
        if ok:
            cache["field"] = f
            _save_cache(cache)
            return f
    raise RuntimeError("No field filter worked; check the OpenAlex documentation.")


def resolve_company_filter(client, field_filter, verbose=True):
    cache = _load_cache()
    if "company" in cache:
        return cache["company"]
    base = f"{field_filter},type:{WORK_TYPE},publication_year:2018"
    total = client.count(base)
    for cand in COMPANY_CANDIDATES:
        try:
            c = client.count(f"{base},{cand}")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [x] {cand:42s} ERROR {str(e)[:70]}")
            continue
        ok = 0 < c < total
        if verbose:
            print(f"  [{'ok' if ok else '??'}] {cand:42s} {c:,} with a company / {total:,} total")
        if ok:
            cache["company"] = cand
            _save_cache(cache)
            return cand
    raise RuntimeError("No company filter worked.")


def resolve_no_company_filter(client, field_filter, company_filter, verbose=True):
    """Filter for papers with NO company affiliation.

    Validated against the identity  total = with_company + without_company  for one year.
    If no candidate satisfies it, the negation syntax changed and the download stops: a
    silently wrong negation would contaminate the control group.
    """
    cache = _load_cache()
    if "no_company" in cache:
        return cache["no_company"]
    base = f"{field_filter},type:{WORK_TYPE},publication_year:2018"
    total = client.count(base)
    with_company = client.count(f"{base},{company_filter}")
    for cand in NO_COMPANY_CANDIDATES:
        try:
            c = client.count(f"{base},{cand}")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [x] {cand:42s} ERROR {str(e)[:70]}")
            continue
        ok = abs((c + with_company) - total) <= max(5, 0.01 * total)   # 1% tolerance
        if verbose:
            print(f"  [{'ok' if ok else '??'}] {cand:42s} {c:,} + {with_company:,} vs {total:,} total")
        if ok:
            cache["no_company"] = cand
            _save_cache(cache)
            return cand
    raise RuntimeError("No valid negation filter: the with/without split would not add up.")


def resolve_lab_filter(client, field_filter, lab_ids, verbose=True):
    """Filter by a list of labs (not cached: it depends on the ids selected)."""
    ids = "|".join(lab_ids)
    base = f"{field_filter},type:{WORK_TYPE},publication_year:2018"
    total = client.count(base)
    for cand in LAB_CANDIDATES:
        f = cand.format(ids=ids)
        try:
            c = client.count(f"{base},{f}")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [x] {cand.split(':')[0]:40s} ERROR {str(e)[:70]}")
            continue
        ok = 0 < c < total
        if verbose:
            print(f"  [{'ok' if ok else '??'}] {cand.split(':')[0]:40s} {c:,} lab articles in 2018")
        if ok:
            return f
    raise RuntimeError("No lab filter worked.")


def year_filter(field_filter, year, extra=None):
    f = f"{field_filter},type:{WORK_TYPE},publication_year:{year}"
    return f"{f},{extra}" if extra else f


# =============================================================================
# 4. LABS / INSTITUTIONS
# =============================================================================
def search_labs(client, queries=None, per_query=5):
    """Search each name in /institutions and save the candidates for manual review."""
    rows = []
    for q in tqdm(queries or LAB_QUERIES, desc="searching institutions"):
        res = client.get("/institutions", {"search": q, "per_page": per_query})
        for rank, inst in enumerate(res["results"], start=1):
            rows.append({
                "query": q, "rank": rank,
                "institution_id": inst["id"].rsplit("/", 1)[-1],
                "display_name": inst["display_name"],
                "type": inst.get("type"),
                "country_code": inst.get("country_code"),
                "works_count": inst.get("works_count"),
                "homepage": inst.get("homepage_url") or "",
                "ror": inst.get("ror", ""),
                "keep": "",    # <- fill in by hand: 1 = lab / Big Tech, 0 = no
                "group": "",   # <- optional: bigtech / ai_lab / other_company
                "notes": "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(LABS_CSV, index=False, encoding="utf-8-sig")
    return df


def load_selected_labs():
    """Ids with keep == 1 in the manually reviewed CSV (duplicates removed)."""
    if not LABS_CSV.exists():
        raise FileNotFoundError("config/labs_candidates.csv is missing: run 'institutions' first.")
    df = pd.read_csv(LABS_CSV, dtype=str).fillna("")
    return list(dict.fromkeys(df[df["keep"].str.strip() == "1"]["institution_id"]))


# =============================================================================
# 5. COUNTS
# =============================================================================
def counts_by_year(client, field_filter, company_filter, lab_filter=None):
    rows = []
    for y in tqdm(YEARS, desc="counts per year"):
        total = client.count(year_filter(field_filter, y))
        comp = client.count(year_filter(field_filter, y, company_filter))
        row = {"year": y, "cs_articles": total, "with_company": comp,
               "without_company": total - comp,
               "pct_company": round(100 * comp / total, 3) if total else None}
        if lab_filter:
            lab = client.count(year_filter(field_filter, y, lab_filter))
            row["with_lab"] = lab
            row["pct_lab"] = round(100 * lab / total, 3) if total else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_DIR / "counts_cs_by_year.csv", index=False)
    return df


def sanity_check(df, total_openalex):
    """List of problems found (empty = everything consistent)."""
    problems = []
    for _, r in df.iterrows():
        if not r["with_company"] < r["cs_articles"]:
            problems.append(f"{r['year']}: with-company is not a proper subset of the total")
        if r["cs_articles"] >= total_openalex * 0.5:
            problems.append(f"{r['year']}: total suspiciously high (the field filter is not filtering)")
        if not 0 < r["pct_company"] < 50:
            problems.append(f"{r['year']}: implausible company share ({r['pct_company']}%)")
    return problems


# =============================================================================
# 6. STORAGE AND INVENTORY
# =============================================================================
def save_raw_json(records, name):
    path = RAW_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    return path


def load_raw_json(name):
    with open(RAW_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _safe(v):
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v


def to_table(records):
    """Flatten the first level of the JSON (nested values stay as objects)."""
    return pd.json_normalize(records, max_level=1)


def save_parquet(df, name):
    """Parquet needs serializable columns: dicts/lists become JSON text."""
    safe = df.astype(object).apply(lambda col: col.map(_safe))
    path = RAW_DIR / f"{name}.parquet"
    safe.to_parquet(path, index=False)
    return path


def inventory(df):
    """Columns with dtype, nulls, an example value and a leakage flag."""
    def is_leak(c):
        return any(c == k or c.startswith(k + ".") for k in LEAKAGE_COLS)

    def first(c):
        s = df[c].dropna()
        return s.iloc[0] if len(s) else None

    inv = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_null": df.isna().sum(),
        "pct_null": (100 * df.isna().mean()).round(1),
        "example": [first(c) for c in df.columns],
    })
    inv["LEAKAGE"] = ["DO NOT USE" if is_leak(c) else "" for c in inv.index]
    return inv.sort_values(["LEAKAGE", "pct_null"], ascending=[False, True])


# =============================================================================
# 6b. READABLE PREVIEW (a flat CSV to show the project to other people)
# =============================================================================
def flatten_work(w, group=None):
    """Pull the fields that matter out of one Work into flat, readable columns.

    This is a first draft of the feature extraction: the same fields will be used to
    build the modelling table later. Leaky fields are kept here ONLY so the preview
    can show the target; they are prefixed LEAK_ as a reminder.
    """
    auths = w.get("authorships") or []
    insts = [i for a in auths for i in (a.get("institutions") or [])]
    types = [i.get("type") for i in insts]
    companies = sorted({i.get("display_name") for i in insts if i.get("type") == "company"})
    countries = sorted({i.get("country_code") for i in insts if i.get("country_code")})
    topic = w.get("primary_topic") or {}
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}) if isinstance(loc, dict) else {}
    cnp = w.get("citation_normalized_percentile") or {}
    oa = w.get("open_access") or {}
    return {
        # --- identification -------------------------------------------------
        "id": (w.get("id") or "").rsplit("/", 1)[-1],
        "title": (w.get("display_name") or "")[:150],
        "year": w.get("publication_year"),
        "type": w.get("type"),
        "group": group,
        # --- TARGET (leaky: never a predictor) -------------------------------
        "LEAK_cited_by_count": w.get("cited_by_count"),
        "LEAK_is_top_10pct": cnp.get("is_in_top_10_percent"),
        # --- who wrote it ----------------------------------------------------
        "n_authors": len(auths),
        "n_institutions": len({i.get("id") for i in insts if i.get("id")}),
        "n_countries": len(countries),
        "countries": ",".join(countries[:5]),
        "is_international": len(countries) > 1,
        # --- industry --------------------------------------------------------
        "has_company": "company" in types,
        "n_companies": sum(t == "company" for t in types),
        "company_names": "; ".join(companies[:3]),
        "has_education": "education" in types,
        # --- where it was published ------------------------------------------
        "venue": (src.get("display_name") or "")[:80],
        "venue_type": src.get("type"),
        "is_oa": oa.get("is_oa"),
        # --- content ----------------------------------------------------------
        "subfield": ((topic.get("subfield") or {}).get("display_name")),
        "topic": topic.get("display_name"),
        "n_references": len(w.get("referenced_works") or []),
        "has_abstract": bool(w.get("abstract_inverted_index")),
        "title_words": len((w.get("display_name") or "").split()),
    }


def make_preview(client, field_filter, company_filter, no_company_filter,
                 year, n_per_group=250, name="preview"):
    """Small balanced sample (company / non-company) exported to a readable CSV."""
    rows = []
    for group, flt in (("company", year_filter(field_filter, year, company_filter)),
                       ("noncompany", year_filter(field_filter, year, no_company_filter))):
        recs = client.fetch_sample(flt, n_per_group, seed=SEED)
        print(f"  {group}: {len(recs)} papers")
        rows += [flatten_work(w, group) for w in recs]
    df = pd.DataFrame(rows)
    path = PROCESSED_DIR / f"{name}_{year}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")   # utf-8-sig so Excel reads accents
    return path, df


# =============================================================================
# 7. FULL DOWNLOAD (all company papers + a random sample without a company)
# =============================================================================
def probe_record_size(client, field_filter, n=20):
    """Average size of one Work record in KB, to estimate how big the download will be."""
    recs = client.get("/works", {"filter": f"{field_filter},type:{WORK_TYPE}",
                                 "per_page": n})["results"]
    if not recs:
        return None
    return len(json.dumps(recs, ensure_ascii=False).encode("utf-8")) / len(recs) / 1024


def plan_download(client, field_filter, company_filter, n_non_company=NON_COMPANY_PER_YEAR):
    """Volumes, estimated API calls and estimated size, without downloading the corpus."""
    kb = probe_record_size(client, field_filter)
    rows = []
    for y in tqdm(YEARS, desc="planning"):
        total = client.count(year_filter(field_filter, y))
        comp = client.count(year_filter(field_filter, y, company_filter))
        non_comp_avail = total - comp
        target = min(n_non_company, non_comp_avail) if n_non_company > 0 else non_comp_avail
        rows.append({"year": y, "cs_total": total, "company": comp,
                     "non_company_target": target,
                     "calls": -(-comp // 100) + -(-target // 100),
                     "est_MB": round((comp + target) * (kb or 0) / 1024, 1)})
    df = pd.DataFrame(rows)
    totals = {c: df[c].sum() for c in df.columns if c != "year"}
    df = pd.concat([df, pd.DataFrame([{"year": "all", **totals}])], ignore_index=True)
    if kb:
        print(f"\nAverage record size: {kb:.1f} KB "
              f"(probe of 20 records; records with many references are larger)")
    return df


def download_year(client, field_filter, company_filter, no_company_filter, year,
                  n_non_company=NON_COMPANY_PER_YEAR, overwrite=False):
    """Download one year into two files. Skips a file that already exists (resumable)."""
    done = []
    for group, flt, n, random in (
        ("company", year_filter(field_filter, year, company_filter), None, False),
        ("noncompany", year_filter(field_filter, year, no_company_filter), n_non_company,
         n_non_company > 0),   # 0 = download every non-company paper (cursor, no sampling)
    ):
        name = f"{group}_{year}"
        path = RAW_DIR / f"{name}.json"
        if path.exists() and not overwrite:
            print(f"  {name}: already downloaded ({path.stat().st_size / 1e6:.1f} MB), skipped")
            done.append(name)
            continue
        if random:
            recs = client.fetch_sample(flt, n, seed=SEED)
        else:
            target = client.count(flt)
            recs = client.fetch_cursor(flt, target)
        save_raw_json(recs, name)
        print(f"  {name}: {len(recs):,} papers -> {path.name}")
        done.append(name)
    return done


def build_dataset(name="dataset_cs"):
    """Merge every year file into one table, tagging group and year. Reads from disk only."""
    frames = []
    for path in sorted(RAW_DIR.glob("*_20*.json")):
        group, year = path.stem.rsplit("_", 1)
        if group not in ("company", "noncompany"):
            continue
        with open(path, encoding="utf-8") as f:
            recs = json.load(f)
        if not recs:
            continue
        df = to_table(recs)
        df["group"] = group
        df["sample_year"] = int(year)
        frames.append(df)
        print(f"  {path.name}: {len(df):,} rows")
    if not frames:
        raise RuntimeError("No year files in data/raw. Run 'download' first.")
    full = pd.concat(frames, ignore_index=True)
    before = len(full)
    full = full.drop_duplicates(subset="id")      # a paper can appear in both groups
    print(f"\nTotal {before:,} rows -> {len(full):,} after removing duplicate ids")
    path = save_parquet(full, name)
    # Freeze the download date: without it the Colab is not reproducible
    meta = {"downloaded_on": time.strftime("%Y-%m-%d"), "rows": len(full),
            "years": [YEARS[0], YEARS[-1]], "seed": SEED,
            "non_company_per_year": NON_COMPANY_PER_YEAR,
            "filters": _load_cache()}
    (PROCESSED_DIR / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path, full


# =============================================================================
# 8. COMMAND LINE
# =============================================================================
def cmd_subfields(client, args):
    res = client.get("/subfields", {"filter": f"field.id:{CS_FIELD_ID}", "per_page": 200})
    df = pd.DataFrame([{"subfield_id": s["id"].rsplit("/", 1)[-1],
                        "display_name": s["display_name"],
                        "works_count": s.get("works_count")} for s in res["results"]])
    df = df.sort_values("works_count", ascending=False, ignore_index=True)
    df.to_csv(PROCESSED_DIR / "cs_subfields.csv", index=False)
    print(df.to_string(index=False))


def cmd_institutions(client, args):
    df = search_labs(client)
    print(f"{len(df)} candidates saved to {LABS_CSV}")
    print("Review the CSV and fill in 'keep' (1/0) before running 'counts' or the download.")


def cmd_counts(client, args):
    ff = resolve_field_filter(client)
    cf = resolve_company_filter(client, ff)
    lab_f = None
    try:
        ids = load_selected_labs()
        if ids:
            lab_f = resolve_lab_filter(client, ff, ids)
    except FileNotFoundError:
        print("(no reviewed lab list yet: those columns are skipped)")
    df = counts_by_year(client, ff, cf, lab_f)
    print(df.to_string(index=False))
    problems = sanity_check(df, client.total_works())
    print("\nSanity check:", "OK" if not problems else "PROBLEMS")
    for p in problems:
        print("  -", p)


def cmd_sample(client, args):
    ff = resolve_field_filter(client)
    flt = f"{ff},type:{WORK_TYPE},publication_year:{YEARS[0]}-{YEARS[-1]}"
    recs = client.fetch_sample(flt, args.n) if args.random else client.fetch_cursor(flt, args.n)
    name = f"sample_{args.n}"
    print("Downloaded:", len(recs))
    print("JSON:", save_raw_json(recs, name))
    print("Parquet:", save_parquet(to_table(recs), name))


def cmd_inventory(client, args):
    inv = inventory(to_table(load_raw_json(f"sample_{args.n}")))
    inv.to_csv(PROCESSED_DIR / f"inventory_sample_{args.n}.csv")
    print(inv.to_string())


def cmd_ratelimit(client, args):
    """Show the daily budget left on this key (/rate-limit endpoint)."""
    if not client.api_key:
        print("No API key in .env: you are on the keyless budget, for testing only.")
        print("Get a free key at https://openalex.org/settings/api")
        return
    info = client.get("/rate-limit")
    rl = info.get("rate_limit", info)
    for k, v in rl.items():
        print(f"  {k:24s} {v}")


def cmd_preview(client, args):
    ff = resolve_field_filter(client)
    cf = resolve_company_filter(client, ff)
    ncf = resolve_no_company_filter(client, ff, cf)
    path, df = make_preview(client, ff, cf, ncf, args.year, args.n)
    print("\nCSV:", path)
    print("Shape:", df.shape)
    print("\nTop-10% rate by group:")
    print(df.groupby("group")["LEAK_is_top_10pct"].mean().round(3).to_string())
    print("\nMost frequent subfields:")
    print(df["subfield"].value_counts().head(8).to_string())
    print("\nMost frequent companies:")
    comp = df.loc[df["group"] == "company", "company_names"].str.split("; ").explode()
    print(comp[comp != ""].value_counts().head(10).to_string())


def cmd_download(client, args):
    ff = resolve_field_filter(client)
    cf = resolve_company_filter(client, ff)

    plan = plan_download(client, ff, cf, args.non_company)
    print("\n" + plan.to_string(index=False))
    last = plan.iloc[-1]
    print(f"\nEstimated API calls: ~{int(last['calls']):,}"
          f"   |   estimated size: ~{last['est_MB'] / 1024:.1f} GB")
    if args.dry_run:
        print("(dry run: nothing downloaded)")
        return
    if last["est_MB"] > 5000:
        print("\nWARNING: over 5 GB. Consider a smaller --non-company value.")

    ncf = resolve_no_company_filter(client, ff, cf)
    years = [args.year] if args.year else YEARS
    for y in years:
        print(f"\n--- {y} ---")
        download_year(client, ff, cf, ncf, y,
                      n_non_company=args.non_company, overwrite=args.overwrite)
    print("\nDone. Now run:  python src/download.py build")


def cmd_build(client, args):
    path, df = build_dataset()
    print("Parquet:", path)
    print("Shape:", df.shape)
    print(df.groupby(["sample_year", "group"]).size().unstack(fill_value=0).to_string())


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--max-calls", type=int, default=None, help="cap on API calls for this run")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("subfields").set_defaults(fn=cmd_subfields)
    sub.add_parser("institutions").set_defaults(fn=cmd_institutions)
    sub.add_parser("counts").set_defaults(fn=cmd_counts)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--random", action="store_true", help="random sample (sample+seed) instead of cursor")
    s.set_defaults(fn=cmd_sample)
    i = sub.add_parser("inventory")
    i.add_argument("--n", type=int, default=500)
    i.set_defaults(fn=cmd_inventory)
    d = sub.add_parser("download", help="full download: all company papers + a sample without")
    d.add_argument("--dry-run", action="store_true", help="only show volumes and estimated calls")
    d.add_argument("--year", type=int, default=None, help="download a single year")
    d.add_argument("--non-company", type=int, default=NON_COMPANY_PER_YEAR,
                   help="control-group size per year; 0 = every non-company paper (very large)")
    d.add_argument("--overwrite", action="store_true", help="re-download files that already exist")
    d.set_defaults(fn=cmd_download)
    b = sub.add_parser("build", help="merge the year files into one parquet")
    b.set_defaults(fn=cmd_build)
    sub.add_parser("ratelimit", help="daily budget left on this key").set_defaults(fn=cmd_ratelimit)
    pv = sub.add_parser("preview", help="small balanced sample exported as a readable CSV")
    pv.add_argument("--year", type=int, default=2018)
    pv.add_argument("--n", type=int, default=250, help="papers per group")
    pv.set_defaults(fn=cmd_preview)

    args = p.parse_args()
    client = OpenAlexClient(max_calls=args.max_calls)
    source = "download.py" if MY_API_KEY else (".env" if client.api_key else None)
    print("API key:", f"yes (from {source})" if client.api_key else "NO")
    args.fn(client, args)
    print(f"\n[API calls in this run: {client.calls}]")


if __name__ == "__main__":
    main()
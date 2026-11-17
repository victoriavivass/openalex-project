# Is Big Tech taking over AI research?

Final project for **Applied Statistical Learning** (ENSAE).

We predict whether a Computer Science paper reaches the **top 10% of citations**, and
measure how much the involvement of **private labs / Big Tech** adds to that prediction.

- **Source:** OpenAlex API (field *Computer Science*, `type:article`, 2014–2021)
- **Target:** `citation_normalized_percentile.is_in_top_10_percent`
- **Secondary task:** regression on `log(1 + citations)`

---

## 1. Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**API key** (free, from <https://openalex.org/settings/api>). Either:

- paste it into `MY_API_KEY` near the top of `src/download.py`, or
- write `OPENALEX_API_KEY=your_key` in a `.env` file at the project root.

Any command prints which source it used. Blank the key out of `src/download.py`
before sharing the code or submitting the Colab notebook.

---

## 2. Commands

| Command | What it does | Cost |
|---|---|---|
| `python src/download.py ratelimit` | daily budget left on the key | 1 call |
| `python src/download.py subfields` | list the subfields of Computer Science | 1 call |
| `python src/download.py institutions` | search lab names → `config/labs_candidates.csv` | ~79 calls |
| `python src/download.py counts` | papers per year, with and without a company | ~30 calls |
| `python src/download.py preview --year 2018` | small balanced sample → readable CSV | ~6 calls |
| `python src/download.py download --dry-run` | volumes, estimated calls and size | ~20 calls |
| `python src/download.py download` | the full download (resumable) | ~1,240 calls |
| `python src/download.py build` | merge the year files into one parquet | 0 (local) |
| `python src/download.py sample --n 500` | raw sample for inspection | ~5 calls |
| `python src/download.py inventory --n 500` | columns and nulls of that sample | 0 (local) |

Global option: `--max-calls N` (before the subcommand) caps API calls for one run.

Between `institutions` and `counts`, open `config/labs_candidates.csv` and fill in the
`keep` column (1/0) and optionally `group` (`bigtech` / `ai_lab` / `other_company`).
That decision is ours and has to be justified in the report.

---

## 3. Project layout

```
openalex-project/
├── config/
│   ├── labs_candidates.csv       # lab search results, reviewed by hand
│   └── filters_resolved.json     # validated OpenAlex filter syntax
├── data/
│   ├── raw/                      # untouched JSON from the API, one file per year+group
│   └── processed/                # tables, counts, previews, the final parquet
├── src/download.py               # everything OpenAlex-related
├── requirements.txt
└── .env                          # optional, holds the API key, never committed
```

`data/raw/` is **never edited**. If the cleaning goes wrong we redo it from the raw
files instead of downloading again.

### Sampling design

For each year 2014–2021:

- **all** papers with at least one `type:company` affiliation (~43,000 in total);
- a **random sample** of 10,000 papers with no company affiliation, drawn with a fixed
  `seed` so it is reproducible.

This oversamples industry papers relative to the real 2.5% share. That is deliberate
(we sample on the features, not on the target, so `P(Y|X)` is preserved), **but it must
be stated in the report**: raw precision and accuracy cannot be read as population
values.

---

## 4. Variables

### 4.1 Preview / modelling columns

These are produced by `flatten_work()` in `src/download.py` (command `preview`).
"Known at publication?" is the column that decides whether a variable may be a predictor.

| Column | Meaning | Source field in OpenAlex | Known at publication? |
|---|---|---|---|
| `id` | OpenAlex work id (e.g. `W2741809807`) | `id` | — (identifier) |
| `title` | title, truncated to 150 characters | `display_name` | yes |
| `year` | year of publication | `publication_year` | yes |
| `type` | document type (always `article` here) | `type` | yes |
| `group` | which sample the paper came from: `company` / `noncompany` | ours | — (design) |
| **`LEAK_cited_by_count`** | **total citations received** | `cited_by_count` | **NO** |
| **`LEAK_is_top_10pct`** | **TARGET: is it in the top 10% of its field and year** | `citation_normalized_percentile.is_in_top_10_percent` | **NO** |
| `n_authors` | number of authors | `len(authorships)` | yes |
| `n_institutions` | number of distinct institutions | distinct `authorships[].institutions[].id` | yes |
| `n_countries` | number of distinct countries | distinct `institutions[].country_code` | yes |
| `countries` | the first 5 country codes, comma separated | same | yes |
| `is_international` | more than one country involved | derived | yes |
| `has_company` | at least one affiliation of type `company` | `institutions[].type` | yes |
| `n_companies` | how many company affiliations | same | yes |
| `company_names` | the first 3 company names | `institutions[].display_name` | yes |
| `has_education` | at least one university affiliation | `institutions[].type` | yes |
| `venue` | journal or conference name | `primary_location.source.display_name` | yes |
| `venue_type` | `journal`, `conference`, `repository`… | `primary_location.source.type` | yes |
| `is_oa` | open access | `open_access.is_oa` | yes |
| `subfield` | CS subfield (AI, Computer Vision, Software…) | `primary_topic.subfield.display_name` | yes |
| `topic` | finer-grained topic | `primary_topic.display_name` | yes |
| `n_references` | number of works cited by this paper | `len(referenced_works)` | yes |
| `has_abstract` | whether an abstract is available | `abstract_inverted_index` | yes |
| `title_words` | number of words in the title | derived | yes |

### 4.2 Variables still to build

Planned for the feature-engineering step, not in the preview yet:

| Variable | How | Note |
|---|---|---|
| `lab_group` | match institution ids against the reviewed `labs_candidates.csv` | `bigtech` / `ai_lab` / `other_company` / `academic_only` |
| `period` | `2014–2017` vs `2018–2021` | the split is fixed **before** looking at results |
| `abstract_length`, TF-IDF features | rebuild the text from `abstract_inverted_index` | fit the vectoriser on the **training set only** |
| `first_author_country`, `n_authors_per_institution` | from `authorships` | OpenAlex lists only the first 100 authors |
| `venue_frequency` | how often the venue appears in the corpus | a proxy for venue prestige, computed on training data only |

### 4.3 Leaky fields — never predictors

These are the target, or derived from what happened to the paper **after** publication.
They are downloaded (we need them to build the label) but excluded from every model.
`LEAKAGE_COLS` in `src/download.py` holds the list, and `inventory` flags them.

`cited_by_count` · `counts_by_year` · `fwci` · `citation_normalized_percentile` ·
`cited_by_percentile_year` · `is_in_top_1_percent` · `related_works` · `updated_date`

---

## 5. What the literature says we should find

### 5.1 Which variables carry the signal

Two review papers converge on the same short list, and **we already have all of it**:

- **Kousha & Thelwall (2024, *JASIST*)** — a review of studies that use regression or
  machine learning to predict citations. The strongest and most universal predictors are
  journal citation rates, the number of authors, properties of the reference list, and
  international collaboration. They also warn that the strength and even the *direction*
  of these associations depends on the set of papers examined, and that there is almost
  never causal evidence.
- **Tahamtan et al. (2016, *Scientometrics*)** — 28 factors in three groups (paper,
  journal, authors). Journal impact factor, international cooperation and number of
  authors correlate most strongly with citations.

Mapped onto our columns: `venue` / `venue_type`, `n_authors`, `n_references`,
`n_countries`, `is_international`.

- **Livne et al. (2013, Microsoft Research)** — machine learning on millions of papers.
  Reference-based features matter most; **content features have little predictive power**.
  Worth knowing before we invest effort in the abstract/TF-IDF block: a weak text result
  is the expected result, not a bug.

### 5.2 Performance benchmark

**Danelakis et al. (2024) reached AUC 0.81** out of sample, with gradient boosting, on
features of this kind; among their most important predictors were the number of pages,
the number of references, and the citations and h-index of the first and last author.

That 0.81 is our **reference point, not a target**: their corpus is headache research,
not Computer Science, so the numbers are not directly comparable. But it tells us the
order of magnitude a well-built model should reach, and that anything near 0.95 means we
have a leakage bug rather than a great model.

### 5.3 Three findings that are interesting whichever way they go

1. **Does `has_company` add anything once venue is controlled for?**
   Both answers are a result:
   - it still adds AUC → there is an industry advantage that better venues and bigger
     teams do not explain;
   - it adds nothing → the Big Tech advantage is **mediated by the venue**: industry
     papers are not cited more for being industry papers, but for appearing at NeurIPS
     with large teams.

   The second outcome sounds like a failure and is the more interesting finding.

2. **Has it changed over time?** Ablation of the industry block per period
   (2014–2017 vs 2018–2021). Our own counts already show company papers nearly doubling
   between 2014 and 2021.

3. **Where does it matter?** Ablation per subfield. A non-uniform effect (strong in AI
   and vision, weak in theory) is itself a result.

### 5.4 Expected risk, decided in advance

**Venue will probably dominate**, and may absorb the company effect. That is the pattern
across the whole literature. To avoid picking the result we like, we commit **now** to
reporting the ablation both with and without the venue block, and discussing both.

### 5.5 Optional extension: author history

Danelakis et al. found first/last-author prior citations among their best predictors. We
do not have them, and they would likely improve the model most.

They can be built from OpenAlex, **but with a trap**: an author's `cited_by_count` is
their count *today*, which includes the citations of the very paper we are predicting —
straightforward leakage. The fix is to use the author's `counts_by_year` and sum only up
to the year *before* publication. That costs one extra API call per author, so we treat
it as an extension, not part of the core.

### References

- Kousha, K. & Thelwall, M. (2024). Factors associating with or predicting more cited or
  higher quality journal articles. *JASIST*, 75(3), 215–244.
- Tahamtan, I., Safipour Afshar, A. & Ahamdzadeh, K. (2016). Factors affecting number of
  citations. *Scientometrics*, 107, 1195–1225.
- Livne, A., Adar, E., Teevan, J. & Dumais, S. (2013). Predicting citation counts using
  text and graph mining. Microsoft Research.
- Danelakis, A. et al. (2024). What predicts citation counts and translational impact in
  headache research? A machine learning analysis. *Cephalalgia*.
- Jurowetzki, R., Hain, D., Mateos-Garcia, J. & Stathoulopoulos, K. (2021). The
  privatization of AI research(-ers). Working paper. Published as *AI & Society* (2025).

---

## 6. Known data limitations

To be stated in the report:

- **Missing affiliation types.** A substantial share of papers have no typed institution,
  so `has_company` is a lower bound, not an exact measurement.
- **Only the first 100 authors** are listed by OpenAlex, which truncates author counts on
  very large collaborations.
- **Citation age.** A 2014 paper has had longer to accumulate citations than a 2021 one.
  The top-10% label is already normalised by field and year; the regression on raw counts
  is not, so year must be controlled for there.
- **Oversampled industry papers**, see the sampling design above.
- **Changing filter syntax.** OpenAlex filters have changed between API versions, and a
  malformed filter can silently return the whole database. Every filter used here is
  validated against a count before use, and the working syntax is cached in
  `config/filters_resolved.json`.

## 7. Reproducibility

`build` writes `data/processed/dataset_cs_metadata.json` with the download date, the
number of rows, the years, the seed, the control-group size and the exact filters used.
OpenAlex is updated daily and citation counts grow, so **results are only reproducible
against that frozen snapshot**. Quote the download date in the report.
## CaiTI-GPT5: Benchmarks and Evaluation Suite (Supplement Repository)

This repository provides a lightweight, reproducible evaluation suite for CaiTI’s core conversational reasoning components, used to benchmark LLMs on tasks aligned with psychotherapeutic workflows. It is a supplement to the paper published on ACM Transactions on Computing for Healthcare and links to the therapist-curated datasets hosted separately.

- Paper: LLM-based Conversational AI Therapist for Daily Functioning Screening and Psychotherapeutic Intervention via Everyday Smart Devices (ACM TOCH) — DOI: [`https://doi.org/10.1145/3712299`](https://doi.org/10.1145/3712299)
- Original dataset repository: `https://github.com/Columbia-ICSL/CaiTI_dataset.git` ([GitHub link](https://github.com/Columbia-ICSL/CaiTI_dataset.git))

### Benchmark tasks

This code operationalizes four benchmark tasks that validate CaiTI’s conversation flow studied in the paper:

- `cbt`: Evaluate whether users correctly proceed through CBT steps (Unhelpful Thoughts → Challenge → Reframe) with binary decisions per stage
- `rv`: Judge if a follow-up response is on-topic/related to the original response/topic (binary)
- `ra37`: Assign a dimension label (among 37) and a score in {0,1,2} for user responses (multiclass)
- `ra_general`: Classify general canonical response classes {Yes, No, Stop, Question, Maybe}

Datasets are curated by licensed psychotherapists and released at `https://github.com/Columbia-ICSL/CaiTI_dataset.git`.

### What’s inside

- `benchmarks/benchmark.py`: single entry to run all tasks with a timestamped output directory
- `benchmarks/config.yaml`: central runtime config (output, caching, parallelism, task config locations)
- `benchmarks/src/`
  - `cbt/`: three-stage Cognitive Behavioral Therapy reasoner evaluation (`cbt.py`, prompts/config)
  - `rv/`: Response Validation (topic/original vs. follow-up relatedness) (`rv.py`, prompts/config)
  - `ra37/`: Response Analyzer over 37 dimensions with 3-level severity score (`ra37.py`, prompts/config)
  - `ra_general/`: General analyzer for yes/no/stop/question/maybe (`ra_general.py`, prompts/config)
  - `openai/`: minimal OpenAI client wrapper and model config (`client_openai.py`, `config_openai.yaml`)
  - `utils/`: logging, IO, cache helpers; 
  - `metrics.py` for accuracy/PR/F1 and confusion matrices
- `benchmarks/environment.yml`: conda environment with pinned versions

`with_dataset` branch include a minimal copy of the CaiTI agent and dataset; Benchmark expects CSVs from the CaiTI dataset repo.


### Quick start

1) Clone this repo and the dataset repo side-by-side (so paths match defaults):

```bash
git clone <this-repo-url> 
cd ./CaiTI_GPT5
git clone https://github.com/Columbia-ICSL/CaiTI_dataset.git
```

Expected structure:

```
<parent>/
  CaiTI_GPT5/
    benchmarks/
      benchmark.py
      config.yaml
      src/
      environment.yml
    CaiTI_dataset/
      ACM HEALTH Datasets_V1 - CBT.csv
      ACM HEALTH Datasets_V1 - RV.csv
      ACM HEALTH Datasets_V1 - Response Analyzer 37 Dimensions.csv
      ACM HEALTH Datasets_V1 - Response Analyzer General.csv
```

2) Create the environment and activate it:

```bash
cd ./benchmarks
conda env create -f environment.yml
conda activate LLM_therapist
```

3) Configure your OpenAI access:

```bash
export OPENAI_API_KEY='YOUR_KEY'
```
or in powershell:
```powershell
$env:OPENAI_API_KEY="YOUR_KEY”
```
Optional: set custom base URL/model via YAML (see below)

4) Run a task (outputs are timestamped under `benchmarks/reports/`):

```bash
# Cognitive Behavioral Therapy (three-stage binary decisions)
python benchmark.py --task cbt

# Response Validation (follow-up relatedness)
python benchmark.py --task rv

# Response Analyzer, 37 dimensions (multiclass with 3-level score)
python benchmark.py --task ra37

# Response Analyzer, general classes (Yes/No/Stop/Question/Maybe)
python benchmark.py --task ra_general
```

### Configuration

- Central runtime config: `benchmarks/config.yaml`
  - `output_root` (default `reports/`), `cache_dir`, `log_dir`
  - `save_intermediate`: persist per-example raw outputs and parsed labels
  - `max_examples`: limit rows for quick trials (set to `null` for full)
  - `parallel`, `parallel_max_batch`: async batching via OpenAI client
  - Task configs: `cbt_config`, `rv_config`, `ra37_config`, `ra_general_config`
- OpenAI model config: `benchmarks/src/openai/config_openai.yaml`
  - `api_base`, `model` (default `gpt-5`), `effort`, `timeout_seconds`
- Task-specific configs (dataset path, CSV column names, prompts):
  - `benchmarks/src/cbt/config_cbt.yaml`
  - `benchmarks/src/rv/config_rv.yaml`
  - `benchmarks/src/ra37/config_ra37.yaml`
  - `benchmarks/src/ra_general/config_ra_general.yaml`

By default, dataset paths point to `../CaiTI_dataset/*.csv` relative to `benchmarks/`. Adjust paths if you place the dataset elsewhere.

### Outputs

For each run, artifacts are saved to `benchmarks/reports/<task>_<YYYYMMDD_HHMMSS>/`:

- `figs/`
  - Confusion matrices per task/stage, e.g. `cm_stage1.png`, `cm_rv.png`, `cm_ra37.png`
- `tables/`
  - Per-example pairs with gold/pred/raw text, e.g. `pairs_stage1.csv`, `pairs_rv.csv`, `pairs_ra37.csv`
  - Summary metrics table, e.g. `metrics.csv`
- `json/`
  - Compact metrics `summary.json`
- `logs/` (under `benchmarks/logs/`): run logs with INFO-level progress

### Caching and reproducibility

- Intermediate results (raw LLM output + parsed labels) are cached in `cache/` if `use_cache: true`
- `cache_max_age_days` purges older entries on startup
- Cache keys derive from the input text and a stable signature of the config/model so changes invalidate old entries

### Data columns expected (from CaiTI_dataset)

- CBT (`ACM HEALTH Datasets_V1 - CBT.csv`):
  - `Statement`, `Unhelpful Thoughts (CBT_Stage1)`, `Unhelpful Thoughts Label`,
    `Challenge (CBT_Stage2)`, `Challenge Label`, `Another Way (CBT_Stage3)`, `Another Way Label`
- RV (`ACM HEALTH Datasets_V1 - RV.csv`):
  - `Dimension`, `Original Response`, `Follow-up Response`, `Follow-up Response Score`
- RA-37 (`ACM HEALTH Datasets_V1 - Response Analyzer 37 Dimensions.csv`):
  - `Dimension`, `Dimension Label`, `Response`, `Score`
- RA-General (`ACM HEALTH Datasets_V1 - Response Analyzer General.csv`):
  - `Dimension`, `Response`

If your CSVs use different headers, update the corresponding `columns` mappings in the task YAMLs under `benchmarks/src/*/`.

### Notes

- This suite uses minimal wrappers over the OpenAI SDK; set `OPENAI_API_KEY` in your environment. You may customize `api_base` and `model` via YAML.
- Progress bars (`tqdm`) and concise logging are enabled; avoid excessive stdout that would break progress bars.
- For quick smoke tests, set `max_examples` in `benchmarks/config.yaml` (e.g., `16`).

### Citation

Please cite and follow the license of the paper if you use this code or the agent or the dataset:

```bibtex
@article{10.1145/3712299,
author = {Nie, Jingping and Shao, Hanya (Vera) and Fan, Yuang and Shao, Qijia and You, Haoxuan and Preindl, Matthias and Jiang, Xiaofan},
title = {LLM-based Conversational AI Therapist for Daily Functioning Screening and Psychotherapeutic Intervention via Everyday Smart Devices},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712299},
doi = {10.1145/3712299},
abstract = {Despite the global mental health crisis, access to screenings, professionals, and treatments remains high. In collaboration with licensed psychotherapists, we propose a Conversational AI Therapist with psychotherapeutic Interventions (CaiTI), a platform that leverages large language models (LLM)s and smart devices to enable better mental health self-care. CaiTI can screen the day-to-day functioning using natural and psychotherapeutic conversations. CaiTI leverages reinforcement learning to provide personalized conversation flow. CaiTI can accurately understand and interpret user responses. When the user needs further attention during the conversation, CaiTI can provide conversational psychotherapeutic interventions, including cognitive behavioral therapy (CBT) and motivational interviewing (MI). Leveraging the datasets prepared by the licensed psychotherapists, we experiment and microbenchmark various LLMs’ performance in tasks along CaiTI's conversation flow and discuss their strengths and weaknesses. With the psychotherapists, we implement CaiTI and conduct 14-day and 24-week studies. The study results, validated by therapists, demonstrate that CaiTI can converse with users naturally, accurately understand and interpret user responses, and provide psychotherapeutic interventions appropriately and effectively. We showcase the potential of CaiTI LLMs to assist the mental therapy diagnosis and treatment and improve day-to-day functioning screening and precautionary psychotherapeutic intervention systems.},
note = {Just Accepted},
journal = {ACM Trans. Comput. Healthcare},
month = jan,
keywords = {Large Language Models (LLMs), Foundation Models, AI therapist, Psychotherapy, Everyday Smart Devices, Cognitive Behavioral Therapy, Motivational Interviewing}
}
```

### License

If not specified elsewhere, this repository is released for research purposes. Please check the dataset repository and other relevant codebase of the paper for its own terms.



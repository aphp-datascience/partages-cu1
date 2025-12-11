# Partages CU1

Temporary repository for PARTAGES CU1

You need to have `uv` installed. Just clone the repo

```bash
git clone https://github.com/aphp-datascience/partages-cu1-preannotation.git
```

Then edit the configuration file to adjust paths, parameters, prompts, Inception schema, etc. (test it on a small subset of your data before launching on the full dataset) and run:

```bash
OPENAI_API_KEY="xxxxxx" uv run scripts/preannotate.py --config config/preannotate_config.yml
```
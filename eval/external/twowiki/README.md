# 2WikiMultiHopQA evaluation

Download [`data_ids_april7.zip`](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1).
Check that its SHA-256 is `95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`.
Extract `dev.json` and `id_aliases.json` into `eval/external/twowiki/raw/`.

From the repository root, prepare the default set of 125 questions per type and suggest an ontology.

```bash
python eval/external/twowiki/prepare.py --data eval/external/twowiki/raw
rag-with-oag suggest eval/external/twowiki/sets/dev500/docs -o eval/external/twowiki/sets/dev500/ontology.yaml
```

Review and edit the suggested `ontology.yaml` before running the evaluation. Replace `<model>`
with the model name.

```bash
python eval/run_golden.py --set eval/external/twowiki/sets/dev500 --kb twowiki --llm <model> --embed bge-m3 --answers --equal-budget
```

The runner skips chain metrics for this set because the evidence entities lack keys that can
be matched to graph facts. An answer must contain an expected alternative. On choice questions,
the runner marks refusals as incorrect. It also marks responses incorrect if a `distractors` entry
appears before the first expected alternative. The report includes the refusal rate for each
question group and retrieval mode.

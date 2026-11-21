# Synthetic Turkish evaluation set

Reproduce this set with the following command.

```bash
python eval/synthetic/generate.py --out eval/synthetic/sets/heldout --seed 101 --variant b --companies 150 --contracts 300 --questions 160
```

- Seed 101
- 593 documents (150 trade registry, 84 annual report, 300 contract, 59 noise)
- 450 entities (150 Sirket, 300 Sozlesme)
- 749 relations (149 ORTAGIDIR, 600 TARAFIDIR)
- 160 questions (40 single_hop, 40 multi_hop, 40 deep_hop, 40 free_text)
- 530,662 document tokens (cl100k_base, counted per file)

Use `--hops 3` with the evaluation runner to allow three graph hops.

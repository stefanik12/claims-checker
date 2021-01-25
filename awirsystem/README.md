# AWIR: Attention-Weighted Information Retrieval system

This is an implementation of information retrieval system weighting the importance of matches
based on selected attention heads' weights.

Attention heads are ranked by their respective importance w.r.t. prediction of keywords 
and keyphrases in the annotated corpora, as documented in notebooks experimenting with 
[head masking](https://github.com/stefanik12/claims-checker/blob/master/notebooks/head_masking.ipynb),
keywords classification [based on heads probing](https://github.com/stefanik12/claims-checker/blob/master/notebooks/head_masking.ipynb),
and based on [attention weights aggregation](https://github.com/stefanik12/claims-checker/blob/master/notebooks/attention_linking.ipynb).

These are inspired by the literature listed in the 
[related lecture of FI:PV212](https://is.muni.cz/el/fi/podzim2020/PV212/index.qwarp?prejit=5595950).

### Evaluation reproduction

Information retrieval on Cranfield Dataset, with selected methodologies of attention heads selection, attention weights aggregation,
distance function, and some other parameters (in config.py and main.py) can be reproduced:

```bash
python -m pip install -r requirements.txt
python main.py
```

Here, AWIRSystem is compared to a weak baseline = random documents ranking, and strong
baseline: Character n-gram TF-IDF system (ranked 11th in last-year 
[Cranfield competition](https://docs.google.com/spreadsheets/d/1f9P3bn17n2rHGCxBnn3GVr57PF5hMWJEILp06Uq7Jnk) 
of FI:PV211).

Currently, AWIR reaches close to 8-10% of Mean Average Precison, depending on configuration.
Random Reaches cca 0.7%, while Character TF-IDF reaches 42% MAP.

## Future work

- Rank and use attentions of system fine-tuned for keywords prediction
- Fine-tune system via token embeddings, to produce compliant contextual embeddings 
  for the same tokens with compliant meaning, while discriminating others (contrastive learning)
- Exact matching of tokens (ablation of contextualized embeddings' matching)

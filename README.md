# AutoEncoder
generated sentence representation by auto encoder:

This simplified demo shows a scheme how to get sentence-level representation in NLP by using unsupervised auto-encoder system. Then, we can adaptively apply the compressed sentence-level feature to downstream tasks (e.g. text classification).
![](imgs/auto_encoder.png)

It's hard to evaluate the rebuilding ability of auto-encoder only to observe *mean square loss*. Therefore, I consider utilizing *cosine similarity* value for better evaluating the performance.
![](imgs/cos_sim.png)

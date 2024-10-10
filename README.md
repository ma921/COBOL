# COBOL: Collaborative Bayesian Optimization with Labelling Experts (COBOL)"

This repository contains the Python code that was presented for the following paper.

[1] Wenjie Xu*, Masaki Adachi*, Colin N. Jones, Michael A. Osborne, Principled Bayesian Optimization in Collaboration with Human Experts. Advances in Neural Information Processing Systems 35 (NeurIPS; **Spotlight**), 2024 <br>
Links: NeurIPS proceedings, arXiv, OpenReview <br>
*: Equal contribution

## Brief explanation
![Animate](./img/concept.png)<br>

**BO-expert collaboration framework**: The algorithm (red) decides if an expert's (blue) label is necessary. If rejected, it generates a different candidate; otherwise, it directly queries.

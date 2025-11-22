---
license: agpl-3.0
language:
- en
datasets:
- alephpi/UniMER-Train
- alephpi/UniMER-Test
- wanderkid/UniMER_Dataset
metrics:
- bleu
- edit-distance
base_model:
- PaddlePaddle/PP-FormulaNet-S
pipeline_tag: image-to-text
tags:
- latex-ocr
---

See more details in https://github.com/alephpi/Texo
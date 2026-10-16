# SEED, не боевой реестр.
# manifest.json = копия origin/sol-ml-analyze:models/synthetic-v2/manifest.json
# (model_key 909185eb..., веса synthetic-v2, обучены на синтетике).
# Настоящие веса и переобученные манифесты положит ml-команда; backend читает
# этот файл как {name, model_key} и отдаёт model_key в /api/manifest.

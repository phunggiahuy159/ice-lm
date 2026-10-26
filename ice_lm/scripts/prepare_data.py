#!/usr/bin/env python
"""Download + tokenize the training mixture (default 70% FineWeb-Edu prose, 30% Python code).

    python scripts/prepare_data.py --out data --train_tokens 2.5e9
    python scripts/prepare_data.py --out data_synth --synthetic          # offline toy corpus for smoke tests
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ice_lm.data import TOKENIZER_DEFAULT, make_synthetic_data, prepare_data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--train_tokens", type=float, default=2.5e9)
ap.add_argument("--val_tokens", type=float, default=2e6, help="per validation split (prose and code)")
ap.add_argument("--code_frac", type=float, default=0.3)
ap.add_argument("--tokenizer", default=TOKENIZER_DEFAULT)
ap.add_argument("--prose_dataset", default="HuggingFaceFW/fineweb-edu")
ap.add_argument("--prose_config", default="sample-10BT")
ap.add_argument("--prose_field", default="text")
ap.add_argument("--code_dataset", default="codeparrot/codeparrot-clean")
ap.add_argument("--code_config", default=None)
ap.add_argument("--code_field", default="content")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

if a.synthetic:
    print(make_synthetic_data(a.out, train_tokens=int(a.train_tokens) if a.train_tokens < 1e8 else 3_000_000,
                              seed=a.seed))
else:
    prepare_data(a.out, int(a.train_tokens), int(a.val_tokens), a.code_frac, a.tokenizer,
                 prose=dict(path=a.prose_dataset, name=a.prose_config, field=a.prose_field),
                 code=dict(path=a.code_dataset, name=a.code_config, field=a.code_field), seed=a.seed)

# streaming datasets leave download threads behind that can block interpreter shutdown (seen on Kaggle);
# everything is written and closed at this point, so exit right away
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)

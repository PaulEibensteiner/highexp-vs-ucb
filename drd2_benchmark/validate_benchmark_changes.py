import numpy as np
import drd2_maxfinding_benchmark as bm
from typing import List, cast

cfg = bm.BenchmarkConfig(
    seed=123,
    pool_size=200,
    seed_size=64,
    rounds=3,
    batch_k=32,
    train_epochs=10,
    train_batch_size=512,
    pred_batch_size=512,
)

out = bm.run_benchmark(cfg)
results = cast(List[bm.MethodResult], out["results"])

print("\nMethod summary:")
for r in results:
    print(
        {
            "method": r.method,
            "best_final": float(r.best_so_far[-1]),
            "simple_regret_final": float(r.simple_regret[-1]),
            "q95": r.threshold_q95,
            "q98": r.threshold_q98,
        }
    )

means = [float(r.best_so_far[-1]) for r in results]
print("\nDistinct final best values:", len(set(np.round(means, 6))))

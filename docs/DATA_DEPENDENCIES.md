# Data Dependencies

This repository includes the minimal ranked-row inputs needed to reproduce the experimental results for the Paper 16 package.

Included files:

- `data/ranked_pools/dev_results.json`
- `data/ranked_pools/dev_reranked_results.json`
- `data/ranked_pools/test_results.json`
- `data/ranked_pools/test_reranked_results.json`

These files are the ranked candidate pools consumed by the reproduction scripts for:

- failure-aware escalation
- robust clarification under noisy answers
- robust repair under noisy edits
- regime-conditioned controller analysis

The repository also includes:

- `data/reference_inputs/controller_summary.json`

This file is a fixed controller summary used by the budget-sweep script. It is included so that the cost-sensitivity analysis can be reproduced without requiring users to rerun the full controller stack.

The paper manuscript itself is intentionally not included in this release.

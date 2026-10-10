# Method Mapping

This implementation follows the central structure of *Learning Multiple Stock
Trading Patterns with Temporal Routing Adaptor and Optimal Transport* (KDD
2021): multiple prediction heads, a router using feature representations and
historical head errors, and an optimal-transport auxiliary routing target.

The repository provides a portable GRU backbone and an explicit label
availability delay for generic factor panels. It is an independent adaptation,
not a copy of Qlib or a claim of the paper's published numerical results.

Paper: https://arxiv.org/abs/2106.12950
Reference implementation: https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA

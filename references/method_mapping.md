# Method Mapping

## Reference

Hengxu Lin, Dong Zhou, Weiqing Liu, and Jiang Bian. *Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport.* KDD 2021. DOI: https://doi.org/10.1145/3447548.3467358

Public paper: https://arxiv.org/abs/2106.12950

## Reproduction status

This repository is an **unofficial independent reproduction and generic factor-panel adaptation**. It is not affiliated with the paper authors or Microsoft Qlib.

## Preserved method structure

- multiple prediction heads for distinct latent patterns;
- routing conditioned on feature representations and historical head errors;
- Gumbel-softmax routing during training;
- Sinkhorn optimal-transport assignment as auxiliary routing supervision.

## Configurable adaptation surface

The package uses a portable GRU backbone and explicit label-availability delay
for generic factor panels. The factor universe, labels, time splits, number of
heads, routing parameters, and optimization settings are configured by the
caller, so the package does not target the paper's particular data or reported
numerical results.

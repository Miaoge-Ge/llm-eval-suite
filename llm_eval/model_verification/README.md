# Model verification (planned)

This module is reserved for comparing an endpoint's claimed model with a trusted
reference. No verification engine or identity score is implemented yet.

Future work can collect reference responses under controlled settings, compare
behavior and output distributions, and report evidence with uncertainty.
Provider model labels, self-identification, one prompt, or a benchmark score
cannot prove that a particular set of model weights served the request.

The first two modules, `benchmark` and `api_testing`, are implemented and can
supply measurements. Their results must not be presented as identity proof.

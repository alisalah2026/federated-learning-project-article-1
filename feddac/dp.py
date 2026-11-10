"""Formal record-level (epsilon, delta)-differential privacy via DP-SGD.

Replaces the heuristic parameter-perturbation of the original submission (old Eq. 11), which
did not use delta, had no bounded sensitivity and no privacy accounting.

Guarantee.  Each client runs DP-SGD (Abadi et al., 2016) on its own data: Poisson subsampling
with rate q_i = B / n_i, per-example gradient clipping to L2 norm Cclip, Gaussian noise with
standard deviation sigma_i * Cclip.  sigma_i is calibrated with the Renyi-DP accountant (Mironov,
2017; Opacus implementation) so that, after all R rounds x (1 / q_i) steps, the client's released
sequence of models is (eps_sgd, delta)-DP with respect to the addition/removal of one training
record.  Because every downstream operation (sequential hand-over, cluster aggregation, global
averaging) is post-processing of DP outputs and clients hold disjoint data, the same guarantee
holds for every method evaluated (FedAvg, FedProx, PP-CFL, FedSe, FedSeq, FedDAC) irrespective of
its aggregation topology, which makes the comparison privacy-equivalent.

Methods that also consume label statistics (DAC, FedSe, FedSeq, PP-CFL) pay for them explicitly:
histograms are released once with the Laplace mechanism (eps_hist, sensitivity 1) and eps_sgd is
reduced accordingly, so that eps_total = eps_sgd + eps_hist is identical across methods
(basic sequential composition).

Public capacity (resubmission protocol).  Under add/remove adjacency the mechanism must not depend
on the private client size n_i.  With `capacity=S_i` (a public bound, n_i <= S_i) the sampling rate
q_i = B / S_i, the number of steps and sigma_i are functions of S_i only; Poisson sampling runs over
the n_i stored records and the noisy sum is divided by the fixed B, so a neighbouring dataset with
n_i +/- 1 records is processed by the same (q_i, T_i, sigma_i).  The historical runs had n_i = S_i
for every client, so they coincide with this mechanism (analysis/plan_checks/stored_sigma_check.json).
The guarantee is per stored record; it covers original examples only if no example is stored twice
(feddac.data.partition(replacement=False)).
"""
from __future__ import annotations

import math


def calibrate_sigma(target_eps: float, delta: float, sample_rate: float, steps: int) -> float:
    from opacus.accountants.utils import get_noise_multiplier

    return float(get_noise_multiplier(target_epsilon=target_eps, target_delta=delta,
                                      sample_rate=min(sample_rate, 1.0), steps=steps,
                                      accountant="rdp", epsilon_tolerance=0.01))


def spent_epsilon(sigma: float, delta: float, sample_rate: float, steps: int) -> float:
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    acc.history = [(sigma, min(sample_rate, 1.0), steps)]
    return float(acc.get_epsilon(delta=delta))


class ClientDP:
    """Per-client DP-SGD configuration."""

    def __init__(self, n_samples: int, batch_size: int, rounds: int, local_epochs: int,
                 eps_sgd: float, delta: float, clip: float = 1.0, capacity: int | None = None):
        if capacity is not None:
            assert n_samples <= capacity, f"client holds {n_samples} records > declared capacity {capacity}"
        self.n_basis = int(capacity if capacity is not None else n_samples)
        self.q = min(1.0, batch_size / max(self.n_basis, 1))
        self.steps_per_round = max(1, int(math.ceil(local_epochs / self.q)))
        self.total_steps = self.steps_per_round * rounds
        self.clip = clip
        self.delta = delta
        self.batch_size = batch_size
        self.eps_target = eps_sgd
        self.sigma = calibrate_sigma(eps_sgd, delta, self.q, self.total_steps)

    def report(self):
        return {"q": self.q, "sigma": self.sigma, "steps": self.total_steps, "n_basis": self.n_basis,
                "eps": spent_epsilon(self.sigma, self.delta, self.q, self.total_steps)}

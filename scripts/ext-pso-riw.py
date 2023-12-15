"""
    PSO-RIW Implementation
"""

max_fevals = 40000
benchmark = "cec_2020"
ND = 50
NA= 40
seeds = 50

from hybris.optim import ParticleSwarm
from hybris.problems import get_benchmark
from hybris.functions import available_functions, test_function
import numpy as np

bench = get_benchmark(benchmark)["problems"]
problems = [ available_functions[fn["function"]] for fn in bench ]
for prob, pd in zip(problems, bench):
    prob.lower = pd["lower"]
    prob.upper = pd["upper"]

def bench_riwpso(prob, seed):
    opt = ParticleSwarm(NA, [ND, 0], max_fevals=max_fevals)
    opt.disable_all_rules()
    opt.vmin[:] = prob["lower"]
    opt.vmax[:] = prob["upper"]
    opt.weights[6,:] = 1.0 # Fully connected
    opt.weights[5,:] = 1.0 # No limit on speed
    opt.reset(int(432*seed**1.5+3213215))
    while not opt.stop():
        x = opt.ask()
        y = test_function(x, name=prob["function"])
        opt.tell(y)
        opt.weights[0] = np.random.rand() * 0.5 + 0.5
    return opt.profile.copy()


if __name__ == "__main__":
    perfs = list()
    for problem_id, (prob, pd) in enumerate(zip(problems, bench)):
        for seed in range(seeds):
            perfs.append(bench_riwpso(pd, seed))
    perfs = np.asarray(perfs).reshape(len(problems), seeds, 1000)
    
    np.save(f"db/tables/{benchmark}_{ND}_{max_fevals}_{'riw'}.npy", perfs)

"""
    APSO-VI Implementation
"""

max_fevals = 40000
benchmark = "harrison_2016"
ND = 40
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

def bench_sapsoiv(prob, seed):
    opt = ParticleSwarm(40, [ND, 0], max_fevals=max_fevals)
    opt.disable_all_rules()
    opt.vmin[:] = prob["lower"]
    opt.vmax[:] = prob["upper"]
    opt.reset(int(432*seed**1.5+3213215))
    opt.weights[6] = 1.0 # Full connectivity
    opt.weights[5] = 1.0 # No upper speed limit (domain size)
    max_iterations = float(max_fevals / 40)

    iteration = 0
    vs = (opt.vmax[0]-opt.vmin[0]) / 2
    while not opt.stop():
        x = opt.ask()
        opt.weights[-2] = 1.0
        y = test_function(x, name=prob["function"])
        opt.tell(y)

        # Empirical mean speed
        vbar = np.abs(opt.speed).mean()

        # Ideal mean speed
        ideal_vbar = vs * (1+np.cos(np.pi*iteration/(0.95*max_iterations))) / 2

        # Campare and in.de.crement
        if vbar < ideal_vbar:
            opt.weights[0] = np.minimum(opt.weights[0] + 0.1, 0.9)
        elif vbar >= ideal_vbar:
            opt.weights[0] = np.maximum(opt.weights[0] - 0.1, 0.3)

        iteration += 1
    return opt.profile[-1]

if __name__ == "__main__":
    perfs = list()
    for problem_id, (prob, pd) in enumerate(zip(problems, bench)):
        for seed in range(seeds):
            perfs.append(bench_sapsoiv(pd, seed))
    perfs = np.asarray(perfs).reshape(len(problems), seeds)
    
    np.save(f"db/tables/{benchmark}_{ND}_{max_fevals}_{'sapso2-vi'}.npy", perfs)

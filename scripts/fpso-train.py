from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-mask_int", type=int, default=None)
parser.add_argument("-mask", type=str, default=None)
parser.add_argument("-max_workers", type=int, default=1)
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-single", action="store_true")
parser.add_argument("-more", action="store_true")
args = parser.parse_args()

mask = args.mask
if not mask:
    mask = f"{args.mask_int:07b}"

if args.single:
    if sum(map(int, list(mask))) > 1:
        exit(0)

if args.more:
    if sum(map(int, list(mask))) < 3:
        exit(0)


import numpy as np
import sys
sys.path.append(".")
from hybris.problems import get_benchmark
from hybris.meta import optimize_self
from hybris.profiling import profile_configurations
import matplotlib.pyplot as plt

benchmark_name = "harrison_2016"
ND = 40
NA = 40
NE = 40000
dbname=f"db/{benchmark_name}_{ND}.npy"

num = 0
num_fuzzy_hyperparams = mask.count("1")
optimizer_args_update = {
        "max_fevals": NE, 
        "num_variables": [ND,0], 
        "num_agents": NA ,
        #"initial_weights": [0.7298, 1.49618, 1.49618,0.0, -16, 0.6, 0.35 ]
        }
prof, bc, etc, new_db = optimize_self(mask, 
        args.seed, num_agents=40, 
        profiler_args={ 
            "max_workers": args.max_workers, 
            "benchmark":benchmark_name,
            "optimizer_args_update": optimizer_args_update
        },
        return_all=True, db=dbname, max_fevals=40*300,
        optimize_membership=True)

#np.save(dbname, new_db)
bc = np.asarray(bc)
memberships, rules = bc[0:2*num_fuzzy_hyperparams].astype(float), bc[2*num_fuzzy_hyperparams:].astype(int)
np.savez(f"db/mopts/mopt_{mask}_{args.seed}.npz", profile=prof, rules=rules, memberships=memberships, mask=mask, opt_args=[NA,ND,NE])



from hybris.meta import expandw
ncont = mask.count("1")*2
res = profile_configurations(
    [
        ([], "0000000", None ),
        ((bc[ncont:]).astype(int), mask, expandw(mask, bc[:ncont]), )
     ], benchmark =  benchmark_name, optimizer_args_update=optimizer_args_update)


fig, axs = plt.subplots(4, 5, figsize=(12,10), dpi=200)
bench = get_benchmark(benchmark_name)["problems"]
axs = axs.flatten()
for i, (ax, name) in enumerate(zip(axs[:19], bench)):
    ax.plot(res[i, 0].mean(0), label="PSO")
    ax.plot(res[i, 1].mean(0), label="FPSO")
    ax.legend()
    ax.text(0.1, 0.9, f"{name['function']}:{round(res[i,0].mean(0)[-1], 2)}/{round(res[i,1].mean(0)[-1], 2)}", transform=ax.transAxes)

fig.savefig(f"./pngs/performance_{mask}_{args.seed}.png")

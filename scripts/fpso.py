from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-mask", type=str, required=True)
parser.add_argument("-max_workers", type=int, required=True)
parser.add_argument("-dims", type=int, required=True)
parser.add_argument("-bench", type=str, required=True)
args= parser.parse_args()
from hybris.profiling import profile_configurations
from hybris.meta import expandw 
import numpy as np
engine = np.load(f"db/mopts/mopt_{args.mask}_4242.npz")
mask = str(engine["mask"])
ND = args.dims
max_fevals = 40000
seeds = 50
NA = 40
benchmark = args.bench
print(mask)
res = profile_configurations([
    ([], "0000000", []),
    (engine["rules"], mask, expandw(mask, engine["memberships"]), )
], benchmark=benchmark, nruns=seeds, 
optimizer_args_update={"num_agents": NA, "max_fevals":max_fevals, "num_variables": [ND,0],
    #"initial_weights": [0.7298, 1.49618, 1.49618,0.0, -16, 0.6, 0.35 ]
    }, 
max_workers=args.max_workers)
# print(res.shape)
# print(expandw(mask, engine["memberships"]))
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(3,4)
# soluces =  [ 100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500 ]
# for i, (ax, r) in enumerate(zip(axs.flat, res)):
#     res = r.mean(axis=1)
#     ax.plot(res[0, :])
#     ax.plot(res[1, :])
#     print(f"{i}: {res[0,-1]:.2E}; {res[1,-1]:.2E}")
# plt.savefig("pngs/A_prof.png")

#np.save(f"db/tables/{benchmark}_{ND}_{max_fevals}_{'ciw'}.npy", res[:, 0, :, -1])
np.save(f"db/tables/{benchmark}_{ND}_{max_fevals}_{mask}.npy",  res[:, 1, :, -1])
np.save(f"db/profiles/fpso_{args.mask}_{benchmark}_{ND}_{max_fevals}.npy",  res[:, :, :, :])

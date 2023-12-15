from argparse import ArgumentParser

# First and foremost

parser = ArgumentParser()
parser.add_argument("-method", required=True, type=str)
parser.add_argument("-seeds", required=True, type=int)
parser.add_argument("-dimensions", required=True, type=int)
parser.add_argument("-max_fevals", required=True, type=int)
parser.add_argument("-benchmark", required=True, type=str)
args = parser.parse_args()

from hybris.problems import get_benchmark
from hybris.functions import available_functions, test_function
import numpy as np
from functools import partial
from tqdm import tqdm

dimensions = args.dimensions
max_fevals = args.max_fevals
problems = get_benchmark(args.benchmark)["problems"]

gprofile = []

if args.method == "GA":
    import geneticalgorithm.geneticalgorithm as ga
    popsize = 50
    max_iters = max_fevals // popsize

    algorithm_param = {'max_num_iteration': max_iters,
                       'population_size':   popsize,
                       'mutation_probability':0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type':'uniform',
                       'max_iteration_without_improv':None}
    
    profiles = np.empty((len(problems), args.seeds, max_iters+1))
    for fi, pd in enumerate(tqdm(problems)):
        bounds = np.array([[pd["lower"], pd["upper"]]]*dimensions)
        func = partial(test_function, name=pd["function"])
        for i in range(args.seeds):
            model = ga(function=func, dimension=dimensions, variable_type='real', variable_boundaries=bounds,
                       progress_bar=False, convergence_curve=False, algorithm_parameters=algorithm_param)
            model.run()
            profiles[fi, i, :] = np.asarray(model.report)
    np.savez_compressed(f"db/{args.benchmark}_{args.dimensions}_{args.max_fevals}_ga.npz", profiles)
    np.save(f"db/tables/{args.benchmark}_{args.dimensions}_{args.max_fevals}_ga.npy", np.min(profiles, axis=-1))


#
# Differential evolution
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
#
 
elif args.method == "DE":
    import copy
    from scipy.optimize import differential_evolution
    popsize = 20
    profiles = []
    for fi, f in enumerate(problems):
        profiles.append([])
        print(f"Processing problem: {f.name}")
        bounds = [(f.lower, f.upper)]*dimensions
        for si in range(args.seeds):
            # fevals = (maxiter + 1) * popsize * len(x)
            maxtiter = int(max_fevals / dimensions / popsize) - 1
            result = differential_evolution(partial(func_wrapper,func=f.func), bounds, maxiter=maxtiter, popsize=popsize, polish=False, tol=0.0)
            #profiles[fi, i, :] = np.asarray(model.report)
            profiles[fi].append(copy.copy(gprofile))
            gprofile.clear()
    profiles = np.asarray(profiles).reshape(len(problems), args.nruns, -1)
    np.save(f"{args.output}.npy", profiles)

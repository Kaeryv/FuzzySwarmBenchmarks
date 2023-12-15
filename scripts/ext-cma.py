import cma
from hybris.problems import get_benchmark
from hybris.functions import available_functions, test_function
from functools import partial
import numpy as np
from copy import copy
import argparse

def bench_cma(problem_data, dimensions, seeds, max_fevals):
    popsize = { 40: 15, 50: 15}
    maxiter = max_fevals // popsize[dimensions]
    fitness_archive = np.full((len(problem_data), seeds, maxiter), np.nan)
    fevals = np.full((len(problem_data), seeds,  maxiter), np.nan)
    for j, pd in enumerate(problem_data):
        func = partial(test_function, name=pd["function"])
        for seed in range(seeds):
            instance_fevals = 0
            es = cma.CMAEvolutionStrategy(dimensions * [0], 0.6, 
                                          {
                                              'BoundaryHandler': cma.s.ch.BoundTransform,
                                              'bounds': [[-1.0], [1.0]],
                                              'maxiter': maxiter,
                                              'tolfun': 0.0,
                                              'tolx': 0.0,
                                              'tolstagnation': np.inf
                                        })
            i = 0
            while not es.stop() and instance_fevals < max_fevals:
                solutions = es.ask()
                st = np.asarray(solutions).copy()
                # solutions \in [-1,1]
                st += 1 # [0, 2]
                st /= 2 # [0, 1]
                st *= (pd["upper"] - pd["lower"])
                st += pd["lower"]
                fitness = [func(x)[0] for x in st]
                es.tell(solutions, fitness)
                fitness_archive[j, seed, i] = np.min(fitness)
                fevals[j, seed, i] = len(fitness)
                instance_fevals+= len(fitness)
                i +=1


    perf = np.nanmin(fitness_archive, axis=-1)
    return perf, fevals, fitness_archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSHADE Benchmarking Script")
    parser.add_argument("-benchmark", type=str, default="harrison_2016", help="Name of the benchmark function")
    parser.add_argument("-dimensions", type=int, default=40, help="Number of dimensions")
    parser.add_argument("-max_evals", type=int, default=40000, help="Maximum number of function evaluations")
    parser.add_argument("-seeds", type=int, default=50, help="Number of random seeds")

    args = parser.parse_args()

    problem_data = get_benchmark(args.benchmark)["problems"]
    perf, fevals, prof = bench_cma(problem_data, args.dimensions, args.seeds, args.max_evals)
    print(prof.shape, fevals.shape)
    np.savez_compressed(f"db/profiles/cma2_{args.benchmark}_{args.dimensions}_{args.max_evals}.npz", profiles=prof, evals=fevals)
    np.savez_compressed(f"db/cma2_{args.benchmark}_{args.dimensions}_{args.max_evals}.npy", perf)





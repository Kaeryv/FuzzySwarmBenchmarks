import sys
sys.path.append(".")
import argparse
import numpy as np
from hybris.functions import available_functions, test_function
from hybris.problems import get_benchmark
from functools import partial
import matplotlib.pyplot as plt
from shady.lshade import optimize_lshade

def run_lshade(problem_data, dimensions, seeds, max_fevals):
    fitness_archive = []
    evals_archive = []

    for pd in problem_data:
        def eval_function(x, name):
            y = test_function(np.ascontiguousarray(x), name=name)
            fitness_archive.append(np.min(y))
            evals_archive.append(len(y))
            return y

        func = partial(eval_function, name=pd["function"])
        
        for seed in range(seeds):
            bounds = np.zeros((2, dimensions))
            bounds[0] = pd["lower"]
            bounds[1] = pd["upper"]
            optimize_lshade(func, bounds, max_fevals)

    return np.asarray(fitness_archive).reshape(len(problem_data), seeds, -1), np.asarray(evals_archive).reshape(len(problem_data), seeds, -1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSHADE Benchmarking Script")
    parser.add_argument("-benchmark", type=str, default="harrison_2016", help="Name of the benchmark function")
    parser.add_argument("-dimensions", type=int, default=40, help="Number of dimensions")
    parser.add_argument("-max_evals", type=int, default=40000, help="Maximum number of function evaluations")
    parser.add_argument("-seeds", type=int, default=50, help="Number of random seeds")

    args = parser.parse_args()

    

    benchmark = args.benchmark
    max_fevals = args.max_evals
    seeds = args.seeds

    problem_data = get_benchmark(benchmark)["problems"]

    profiles, evals = run_lshade(problem_data, args.dimensions, seeds, max_fevals)
    np.save(f"db/{benchmark}_{args.dimensions}_{max_fevals}_{'lshade'}.npy", np.min(profiles, axis=-1))
    np.savez_compressed(f"db/profiles/lshade_{args.benchmark}_{args.dimensions}_{max_fevals}.npz", profiles=profiles, evals=evals)

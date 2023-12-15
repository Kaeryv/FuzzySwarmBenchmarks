"""
    BBOB Benchmark
    Script designed for array job.
    Specify the number of job with -batches and the job id with -batch.
"""
w = 0.792
c1 = 1.4944
c2 = 1.4944
popsize = 40

from sko.PSO import PSO

from hybris.meta import expandw 
from hybris.optim import ParticleSwarm
from hybris import Parameter

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from math import ceil

import cocoex
import cocopp

def fmin_hybris(FUN, DIM, maxfunevals):
  engine = np.load(args.fuzzy_engine) if args.fuzzy_engine else None
  remaining_evals = copy(maxfunevals)
  while remaining_evals > popsize:
    opt = ParticleSwarm(popsize, [DIM, 0], max_fevals=remaining_evals)
    opt.vmin = -5.0 # Standard boundaries for all BBOB problems
    opt.vmax =  5.0
    if engine: # If a fuzzy controller is given, use it.
        mask = str(engine["mask"])
        opt.set_rules_fromlist(mask, engine["rules"])
        opt.set_memberships(mask, expandw(mask, engine["memberships"]))
    # Randomized seed
    opt.reset(np.random.randint(99999))
    while not opt.stop():
      x = opt.ask()
      y = np.ascontiguousarray([ FUN(xi) for xi in x ])
      remaining_evals -= popsize
      opt.tell(y)
      # Restart strategy is based on stagnation detection.
      # If there is no stagnation, the optimizer can go on for "remaining_evals".
      if opt.handle.contents.si > args.max_stagnation:
         print("STAG")
         break


# PSO from scikit-optimize
def fmin_sko(FUN, DIM, maxfunevals):
  pso = PSO(func=FUN, n_dim=DIM, pop=popsize, max_iter=ceil(maxfunevals/popsize), 
        lb=-5 * np.ones((DIM)), ub=5*np.ones((DIM)), w=w, c1=c1, c2=c2)


# PSO from bbob archive for comparison
# We boosted it with the same simple restart strategy
# The code is kept identical to the matlab version PSO_el_ab
def fmin_kamel(FUN, DIM, maxfunevals):
    remaining_evals = copy(maxfunevals)
    while remaining_evals > popsize:
      stagnation_time = 0
      current_best = np.inf
      # Set algorithm parameters
      xbound = 5
      vbound = 5
      # Allocate memory and initialize
      xmin = -xbound * np.ones((popsize,DIM))
      xmax = xbound * np.ones((popsize,DIM))
      vmin = -vbound * np.ones((popsize,DIM))
      vmax = vbound * np.ones((popsize,DIM))
      x = 2 * xbound * np.random.rand(popsize,DIM) - xbound;
      v = 2 * vbound * np.random.rand(popsize,DIM) - vbound;
      pbest = x.copy()
      # update pbest and gbest
      cost_p = np.asarray([FUN(xx) for xx in pbest])
      cost, index = np.min(cost_p), np.argmin(cost_p)
      gbest = pbest[index,:].copy()
      maxiterations = ceil(remaining_evals/popsize)
      for it in range(maxiterations):
          # Update velocity
          v = w*v + c1*np.random.rand(popsize,DIM)*(pbest-x) + c2*np.random.rand(popsize,DIM)* (gbest-x)
  
          # Clamp veloctiy
          s = v < vmin
          v = (1-s).astype(float) * v + s * vmin
          b = v > vmax
          v = (1-b).astype(float) * v + b * vmax
  
          # Update position
          x += v
  
          # Clamp position - Absorbing boundaries
          # Set x to the boundary
          s = x < xmin # bool
          x = (1-s).astype(float)*x + s.astype(float) * xmin
          b = x > xmax
          x = (1-b).astype(float)*x + b.astype(float) * xmax
  
          # Clamp position - Absorbing boundaries
          # Set v to zero
          b = np.logical_or(s, b)
          v = (1-b)*v + b*np.zeros((popsize,DIM))
          # Update pbest and gbest if necessary
          cost_x = np.asarray([FUN(xx) for xx in x])
          remaining_evals -= popsize
          s = cost_x < cost_p
          cost_p[s] = cost_x[s]
          pbest[s, :] = x[s]
          cost, index = np.min(cost_p), np.argmin(cost_p)

          # Our addition: 
          if cost < current_best:
              current_best = copy(cost)
              stagnation_time = 0
          else:
              stagnation_time += 1
          gbest = pbest[index,:]
          if stagnation_time >= args.max_stagnation:
              print("STAG", it)
              break
          # End of addition

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument("-batch", type=int)
  parser.add_argument("-batches", type=int, default=600)
  parser.add_argument("-name", type=str)
  parser.add_argument("-fev", type=float)
  parser.add_argument("-algo", type=str)
  parser.add_argument("-fuzzy_engine", type=str, default=None)
  parser.add_argument("-max_stagnation", type=int, default=20)
  args = parser.parse_args()
  
  assert(args.algo in ["kamel", "sko", "hybris"])
  

  
  
  batches = args.batches
  current_batch = args.batch if args.batch else 1
  output_folder = f"{args.name}/"
  if batches > 1:
      output_folder += "batch%03dof%d" % (current_batch, batches)
  
  budget_multiplier = int(args.fev) # times dimension, increase to 10, 100, ...
  suite = cocoex.Suite("bbob", "year: 2009", "dimensions: 2,3,5,10,20,40")
  observer = cocoex.Observer("bbob", "result_folder: " + output_folder)
  print("Suite length", len(suite))
  for batch_counter, problem in enumerate(suite):
      if batch_counter % batches != current_batch % batches:
          continue
      problem.observe_with(observer)  # generate the data for cocopp post-processing
      problem(np.zeros(problem.dimension))  # making algorithms more comparable
      propose_x0 = problem.initial_solution_proposal  # callable, all zeros in first call
      budget = problem.dimension * budget_multiplier
      fmin = globals()[f"fmin_{args.algo}"]
      fmin(problem,  problem.dimension, budget)
      

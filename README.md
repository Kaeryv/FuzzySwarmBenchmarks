## Benchmarking

## Internal benchmarks

| Algorithm   | File                   |Notes   |
|-------------|------------------------|--------|
| APSO-VI     | scripts/ext-apso-vi.py ||
| PSO-RIW     | scripts/ext-pso-riw.py ||
| CMA-ES      | scripts/ext-cma.py     ||
| GA          | scripts/ext-ga-de.py   ||
| L-SHADE     | scripts/ext-shade.py   | We used our own implementation instead of PyADE because it wasn't correct (bad performance)|
| FPSO        | scripts/fpso.py        ||

## BBOB Benchmark

- scripts/bbob.py

## Set of controllers

A sample of trained controllers is provided in folder `controllers`.
They can be used in the FPSO benchmark script or analyzed any way you want.

## Dataset

| Path       | In the article |                                   | Filename meaning                     |
| db/tables/ | Tables 7 & 8   | Used for accuracy and consistency | {bench}_{dims}_{fevals}_{method}.npz |

## Training new controllers

This can be done using `scripts/fpso-train.py`.

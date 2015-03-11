Conway\'s Game of Life
======================

CUDA 2.0 implementation of Conway's Game of Life.

Rules
=====

According to Wikipedia:

> The Unverse of the Game of Life is an infinite two-dimensional orthogonal grid of square cells, each of which is in one of two possible states, alive or dead. Every cell interacts with its eight neighbours, which are the cells that are horizontally, vertically, or diagonally adjacent. At each step in time, the following transitions occur:

> 1. Any live cell with fewer than two live neighbours dies, as if caused by under-population.
> 2. Any live cell with two or three live neighbours lives on to the next generation.
> 3. Any live cell with more than three live neighbours dies, as if by overcrowding.
> 4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

> The initial pattern constitutes the seed of the system. The first generation is created by applying the above rules simultaneously to every cell in the seed—births and deaths occur simultaneously, and the discrete moment at which this happens is sometimes called a tick (in other words, each generation is a pure function of the preceding one). The rules continue to be applied repeatedly to create further generations.

Rationale
=========

The fact that we can process each cell\'s next state in parallel, independent of its neighbours, makes this problem very suitable for implementation on a GPU.

This program shows how to run the Game of Life on NVIDIA GPUs using CUDA 2.0.


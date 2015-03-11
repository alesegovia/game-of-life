/**
 * A CUDA 2.0 Implementation of the Game of Life.
 *
 * Copyright (C) 2015 Alejandro Segovia
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *		    
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *				    
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define BLOCK_SIDE 16

typedef unsigned char ubyte;

__device__ ubyte getat(ubyte* pboard, int nrows, int ncols, int x, int y)
{
	if (x >= 0 && x < ncols && y >= 0 && y < nrows)
		return pboard[x * ncols + y];
	return 0x0;
}

__device__ int numneighbors(int x, int y, ubyte* pboard, int nrows, int ncols)
{
	int num = 0;

	num += (getat(pboard, nrows, ncols, x-1, y));

	num += (getat(pboard, nrows, ncols, x+1, y));
	
	num += (getat(pboard, nrows, ncols, x, y-1));
	
	num += (getat(pboard, nrows, ncols, x, y+1));
	
	num += (getat(pboard, nrows, ncols, x-1, y-1));
	
	num += (getat(pboard, nrows, ncols, x-1, y+1));
	
	num += (getat(pboard, nrows, ncols, x+1, y-1));
	
	num += (getat(pboard, nrows, ncols, x+1, y+1));
	
	return num;
}

__global__ void simstep(int nrows, int ncols, ubyte* pCurrBoard, ubyte* pNewBoard)
{
	int x = blockIdx.x * BLOCK_SIDE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIDE + threadIdx.y;

	int indx = x * ncols + y;

	pNewBoard[indx] = pCurrBoard[indx];

	int neighbors = numneighbors(x, y, pCurrBoard, nrows, ncols);

	// Apply game rules:
	// "Any live cell with fewer than two live neighbours dies, 
	// as if caused by under-population." [WIK11]
	if (neighbors < 2)
		pNewBoard[indx] = 0x0;

	// "Any live cell with two or three live neighbours lives on 
	// to the next generation." [WIK11]
	// (do nothing)

	// "Any live cell with more than three live neighbours dies, 
	// as if by overcrowding." [WIK11]
	if (neighbors > 3)
		pNewBoard[indx] = 0x0;

	// "Any dead cell with exactly three live neighbours becomes 
	// a live cell, as if by reproduction." [WIK11]
	if (neighbors == 3 && !pCurrBoard[indx])
		pNewBoard[indx] = 0x1;
}

void randomizeBoard(ubyte* pboard, int nrows, int ncols, float probability)
{
	for (int x = 0; x < ncols; x++)
	{
		for (int y = 0; y < nrows; y++)
		{
			float rnd = rand() / (float)RAND_MAX;
			pboard[x * ncols + y] = (rnd >= probability)? 0x1 : 0x0;
		}
	}
}

void printBoard(const char* msg, ubyte* pboard, int nrows, int ncols)
{
	printf("%s\n", msg);

	for (int x = 0; x < ncols; x++)
	{
		for (int y = 0; y < nrows; y++)
		{
			printf("%c ", pboard[x * ncols + y]? 'o' : ' ');
		}
		printf("\n");
	}

}

int main(int argc, char* argv[])
{
	int boardW = 64;
	int boardH = 64;

	int ngenerations = 1000000;
	if (argc > 1)
	{
		ngenerations = atoi(argv[1]);
	}

	printf("Running %d generations\n", ngenerations);

	srand(time(0));

	ubyte* pboard = (ubyte *)malloc(boardW * boardH * sizeof(ubyte));
	randomizeBoard(pboard, boardH, boardW, 0.7f);
	printBoard("Initial Board:", pboard, boardH, boardW);

	ubyte* pDevBoard0;
	cudaMalloc((void **)&pDevBoard0, boardW * boardH * sizeof(ubyte));
	cudaMemcpy(pDevBoard0, pboard, boardH * boardW * sizeof(ubyte), cudaMemcpyHostToDevice);

	ubyte* pDevBoard1;
	cudaMalloc((void **)&pDevBoard1, boardW * boardH * sizeof(ubyte));
	cudaMemset(pDevBoard1, 0x0, boardH * boardW * sizeof(ubyte));

	dim3 blocksize(BLOCK_SIDE, BLOCK_SIDE);
	dim3 gridsize(boardW / BLOCK_SIDE, boardH / BLOCK_SIDE);

	struct timeval ti;
	gettimeofday(&ti, NULL);

	ubyte* pcurr;
	ubyte* pnext;
	for (int gen = 0; gen < ngenerations; gen++)
	{
		if ((gen % 2) == 0)
		{
			pcurr = pDevBoard0;
			pnext = pDevBoard1;
		}
		else
		{
			pcurr = pDevBoard1;
			pnext = pDevBoard0;
		}

		simstep<<<gridsize, blocksize>>>(boardH, boardW, pcurr, pnext);

#ifdef PRINT_BOARDS
		cudaMemcpy(pboard, pnext, boardH * boardW * sizeof(ubyte), cudaMemcpyDeviceToHost);
		for (int i = 0; i < 24; i++) printf("\n");
		printBoard(" ", pboard, boardH, boardW);
		//usleep(250000);
#endif

	}

	struct timeval tf;
	gettimeofday(&tf, NULL);
	double t = ((tf.tv_sec - ti.tv_sec) * 1000.0) + ((tf.tv_usec - ti.tv_usec) / 1000.0);


	cudaMemcpy(pboard, pcurr, boardW * boardH * sizeof(ubyte), cudaMemcpyDeviceToHost);

	printBoard("Resulting Board:", pboard, boardH, boardW);

	cudaFree(pDevBoard0);
	cudaFree(pDevBoard1);
	free(pboard);

	printf("%d generations in %f milliseconds\n", ngenerations, t);

	return 0;
}


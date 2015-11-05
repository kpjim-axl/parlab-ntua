/*
 *  gpu_kernels.cu -- GPU kernels
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Vasileios Karakasis
 */ 

#include <stdio.h>
#include <cuda.h>
#include "error.h"
#include "gpu_util.h"
#include "graph.h"
#include "timer.h"
#include "alloc.h"

#define GPU_KERNEL_NAME(name)   do_apsp_gpu ## name
#define BLOCK_SIZE 32
#define GRID_SIZE n

#define TILE(a,b,i,j) dist[(a*GPU_TILE_DIM+i)*n+(b*GPU_TILE_DIM+j)]
weight_t *copy_graph_to_gpu(const graph_t *graph)
{
    size_t dist_size = graph->nr_vertices*graph->nr_vertices;
    weight_t *dist_gpu = (weight_t *) gpu_alloc(dist_size*sizeof(*dist_gpu));
    if (!dist_gpu)
        error(0, "gpu_alloc() failed: %s", gpu_get_last_errmsg());

    if (copy_to_gpu(graph->weights[0], dist_gpu,
                    dist_size*sizeof(*dist_gpu)) < 0)
        error(0, "copy_to_gpu() failed: %s", gpu_get_last_errmsg());

    return dist_gpu;
}

graph_t *copy_graph_from_gpu(const weight_t *dist_gpu, graph_t *graph)
{
    size_t dist_size = graph->nr_vertices*graph->nr_vertices;

    if (copy_from_gpu(graph->weights[0], dist_gpu,
                      dist_size*sizeof(*dist_gpu)) < 0)
        error(0, "copy_from_gpu() failed: %s", gpu_get_last_errmsg());

    return graph;
}



/***************************************************************************************/
/*****					KERNEL: Naive				   *****/
/***************************************************************************************/
/*
 * The naive GPU kernel
 */

__global__ void GPU_KERNEL_NAME(_naive)(weight_t *dist, int n, int k){
    uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
    
    dist[tidy*n + tidx] = MIN(dist[tidy*n + tidx], dist[tidy*n + k] + dist[k*n + tidx]);
}


/***************************************************************************************/
/*****			KERNEL: Tiled - Global Memory				   *****/
/***************************************************************************************/
/*
 *  The tiled GPU kernel(s) using global memory
 */ 

__device__ void FWI_tiled(weight_t *dist, int a, int b, int c, int d, int e, int f, int n){
    uint k;
    for (k=0; k<GPU_TILE_DIM; k++){
	TILE(a, b, threadIdx.y, threadIdx.x) = MIN( TILE(a, b, threadIdx.y, threadIdx.x), TILE(c, d, threadIdx.y, k) + TILE(e, f, k, threadIdx.x) );
	__syncthreads();
    }
}

__global__ void GPU_KERNEL_NAME(_tiled_stage_1)(weight_t *dist, int n, int k)
{
    FWI_tiled(dist, k,k, k,k, k,k, n);
}

/* duo seires apo blocks. h prwth ypologizei thn k_row kai h deuterh thn k_line */
__global__ void GPU_KERNEL_NAME(_tiled_stage_2)(weight_t *dist, int n, int k)
{
    if (blockIdx.x == k)
	return;
    if (blockIdx.y == 0)
	FWI_tiled(dist, blockIdx.x,k, blockIdx.x,k, k,k, n);
    else
	FWI_tiled(dist, k,blockIdx.x, k,k, k,blockIdx.x, n);
}

__global__ void GPU_KERNEL_NAME(_tiled_stage_3)(weight_t *dist, int n,
                                                int k)
{
    if (blockIdx.x == k)
	return;
    if (blockIdx.y == k)
	return;
    
    FWI_tiled(dist, blockIdx.y,blockIdx.x, blockIdx.y,k, k,blockIdx.x, n);
}





/***************************************************************************************/
/*****			KERNEL: Tiled - Shared Memory				   *****/
/***************************************************************************************/
__device__ void FWI_shared_tiled(weight_t tile_1[][GPU_TILE_DIM], weight_t tile_2[][GPU_TILE_DIM], weight_t tile_3[][GPU_TILE_DIM]){
    uint k;
    for (k=0; k<GPU_TILE_DIM; k++){
	tile_1[threadIdx.y][threadIdx.x] = MIN(tile_1[threadIdx.y][threadIdx.x], tile_2[threadIdx.y][k] + tile_3[k][threadIdx.x]);
	__syncthreads();
    }
}

__device__ void fetch(int a, int b, weight_t tile[][GPU_TILE_DIM], weight_t *dist, int n){
    tile[threadIdx.y][threadIdx.x] = TILE(a,b, threadIdx.y, threadIdx.x);
}

__device__ void send(weight_t tile[][GPU_TILE_DIM], int a, int b, weight_t *dist, int n){
    TILE(a,b, threadIdx.y, threadIdx.x) = tile[threadIdx.y][threadIdx.x];
}

__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_1)(weight_t *dist, int n, int k){
    __shared__ weight_t tile_k[GPU_TILE_DIM][GPU_TILE_DIM];

    fetch(k,k, tile_k, dist, n);
    __syncthreads();
    
    FWI_shared_tiled(tile_k, tile_k, tile_k);
    send(tile_k, k,k, dist, n);
}

/* opws ston tiled-kernel, exoume 2 grammes apo nr_tiles blocks */
__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_2)(weight_t *dist, int n, int k){
    if (blockIdx.x == k)
	return;
    
    __shared__ weight_t tile_k[GPU_TILE_DIM][GPU_TILE_DIM];
    __shared__ weight_t my_tile[GPU_TILE_DIM][GPU_TILE_DIM];
    
    fetch (k,k, tile_k, dist, n);
    __syncthreads();
    
    if (blockIdx.y == 0){
	fetch(blockIdx.x, k, my_tile, dist, n);
	__syncthreads();
	
	FWI_shared_tiled(my_tile, my_tile, tile_k);
	send(my_tile, blockIdx.x, k, dist, n);
    }
    else {
	fetch(k, blockIdx.x, my_tile, dist, n);
	__syncthreads();
	
	FWI_shared_tiled(my_tile, tile_k, my_tile);
	send(my_tile, k, blockIdx.x, dist, n);
    }
}

__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_3)(weight_t *dist, int n, int k){
    if ((blockIdx.x == k) || (blockIdx.y == k))
	return;
    
    __shared__ weight_t tile_ij[GPU_TILE_DIM][GPU_TILE_DIM];
    __shared__ weight_t tile_ik[GPU_TILE_DIM][GPU_TILE_DIM];
    __shared__ weight_t tile_kj[GPU_TILE_DIM][GPU_TILE_DIM];
    
    fetch(blockIdx.y, blockIdx.x, tile_ij, dist, n);
    fetch(blockIdx.y, k, tile_ik, dist, n);
    fetch(k, blockIdx.x, tile_kj, dist, n);
    __syncthreads();
    FWI_shared_tiled(tile_ij, tile_ik, tile_kj);
    send(tile_ij, blockIdx.y, blockIdx.x, dist, n);
}

/*
 *  FILLME: Use different kernels for the different stages of the
 *  tiled FW computation
 *  
 *  Use GPU_TILE_DIM (see graph.h) as the tile dimension. You can
 *  adjust its value during compilation. See `make help' for more
 *  information.
 */ 


/***************************************************************************************/
/*****				CPU: Call Kernels				   *****/
/***************************************************************************************/

/***************************************************************************************/
/*****				Call: Naive Kernel				   *****/
/***************************************************************************************/

graph_t *MAKE_KERNEL_NAME(_gpu, _naive)(graph_t *graph)
{
    xtimer_t transfer_timer;
    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    weight_t *dist_gpu = copy_graph_to_gpu(graph);
    timer_stop(&transfer_timer);

    /* FILLME: Set up and launch the kernel(s) */
    
    int k;
    int n = graph->nr_vertices;
    dim3 block(32, 32);
    dim3 grid(n/32, n/32);
    
    for (k=0; k<n; k++){
	GPU_KERNEL_NAME(_naive)<<<grid, block>>>(dist_gpu, n, k);
	cudaThreadSynchronize();
    }
    
    /*
     * Wait for last kernel to finish, so as to measure correctly the
     * transfer times Otherwise, copy from GPU will block
     */
//     cudaThreadSynchronize();

    /* Copy back results to host */
    timer_start(&transfer_timer);
    copy_graph_from_gpu(dist_gpu, graph);
    timer_stop(&transfer_timer);
    printf("Total transfer times: %lf s\n",
           timer_elapsed_time(&transfer_timer));
    return graph;
}


/***************************************************************************************/
/*****				Call: Tiled Kernel				   *****/
/***************************************************************************************/

graph_t *MAKE_KERNEL_NAME(_gpu, _tiled)(graph_t *graph)
{
    xtimer_t transfer_timer;
    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    weight_t *dist_gpu = copy_graph_to_gpu(graph);
    timer_stop(&transfer_timer);

    uint N = graph->nr_vertices;
    uint nr_tiles = N / GPU_TILE_DIM;
    dim3 block(GPU_TILE_DIM, GPU_TILE_DIM);
    dim3 grid1(1);
    dim3 grid2(nr_tiles, 2);
    dim3 grid3(nr_tiles, nr_tiles);
    uint k;
    
    for (k=0; k<nr_tiles; k++){
	GPU_KERNEL_NAME(_tiled_stage_1)<<<grid1, block>>>(dist_gpu, N, k);
	cudaThreadSynchronize();
	
	GPU_KERNEL_NAME(_tiled_stage_2)<<<grid2, block>>>(dist_gpu, N, k);
	cudaThreadSynchronize();
	
	GPU_KERNEL_NAME(_tiled_stage_3)<<<grid3, block>>>(dist_gpu, N, k);
	cudaThreadSynchronize();
    }
    /*
     * Wait for last kernel to finish, so as to measure correctly the
     * transfer times Otherwise, copy from GPU will block
     */
//     cudaThreadSynchronize();

    /* Copy back results to host */
    timer_start(&transfer_timer);
    copy_graph_from_gpu(dist_gpu, graph);
    timer_stop(&transfer_timer);
    printf("Total transfer times: %lf s\n",
           timer_elapsed_time(&transfer_timer));
    return graph;
}

/***************************************************************************************/
/*****				Call: Tiled ShMem Kernel			   *****/
/***************************************************************************************/
graph_t *MAKE_KERNEL_NAME(_gpu, _tiled_shmem)(graph_t *graph)
{
    xtimer_t transfer_timer;
    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    weight_t *dist_gpu = copy_graph_to_gpu(graph);
    timer_stop(&transfer_timer);

    uint N = graph->nr_vertices;
    uint nr_tiles = N / GPU_TILE_DIM;
    dim3 block(GPU_TILE_DIM, GPU_TILE_DIM);
    dim3 grid1(1);
    dim3 grid2(nr_tiles, 2);
    dim3 grid3(nr_tiles, nr_tiles);
    uint k;
    for (k=0; k<nr_tiles ;k++){
	
	GPU_KERNEL_NAME(_tiled_shmem_stage_1)<<<grid1, block>>>(dist_gpu, N, k);
	cudaThreadSynchronize();
	
	GPU_KERNEL_NAME(_tiled_shmem_stage_2)<<<grid2, block>>>(dist_gpu, N, k);
	cudaThreadSynchronize();
	
	GPU_KERNEL_NAME(_tiled_shmem_stage_3)<<<grid3, block>>>(dist_gpu, N, k);
	cudaThreadSynchronize();
    }
    /*
     * Wait for last kernel to finish, so as to measure correctly the
     * transfer times Otherwise, copy from GPU will block
     */
//     cudaThreadSynchronize();

    /* Copy back results to host */
    timer_start(&transfer_timer);
    copy_graph_from_gpu(dist_gpu, graph);
    timer_stop(&transfer_timer);
    printf("Total transfer times: %lf s\n",
           timer_elapsed_time(&transfer_timer));
    return graph;
}

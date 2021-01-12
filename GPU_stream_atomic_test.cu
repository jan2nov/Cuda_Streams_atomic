#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utils_cuda.h"

#include <omp.h>

#define WARP 32
#define HALF_WARP 16


int device=0;

template<typename T, typename V>
__device__ __inline__ T shfl_down(const T &XX, const V &YY) {
	#if(CUDART_VERSION >= 9000)
		return(__shfl_down_sync(0xffffffff, XX, YY));
	#else
		return(__shfl_down(XX, YY));
	#endif
}

__device__ __inline__ void Reduce_SM(float *sum, float *s_input){
	(*sum) = s_input[threadIdx.x];
	
	for (int i = ( blockDim.x >> 1 ); i > HALF_WARP; i = i >> 1) {
		if (threadIdx.x < i) {
			(*sum) = (*sum) + s_input[threadIdx.x + i];
			s_input[threadIdx.x] = (*sum);
		}
		__syncthreads();
	}
}

__device__ __inline__ void Reduce_WARP(float *sum){
	float l_sum;
	
	for (int q = HALF_WARP; q > 0; q = q >> 1) {
		l_sum = shfl_down((*sum), q);
		__syncwarp();
		(*sum) = (*sum) + l_sum;
	}
}

__global__ void reduce_using_atomics(float *d_output, float const* __restrict__ d_input, unsigned int *d_count, float *d_sum, int nElements_per_stream) {
	float sum;
	__shared__ float s_input[128];
//	__shared__ bool isLastBlockDone;
	
	unsigned long int pos = blockIdx.x*128 + threadIdx.x;
	s_input[threadIdx.x] = d_input[pos];
	sum = 0;
	
	__syncthreads();
	Reduce_SM(&sum, s_input);
	__syncthreads();
	Reduce_WARP(&sum);
	__syncthreads();
	
	if(threadIdx.x == 0) {
		atomicAdd(d_sum, sum);
		__threadfence();
	}
}


__global__ void dummy_copy(float *d_output, float *d_sum){
	d_output[0] = d_sum[0];
//	printf("Thread: %d; value=%f;\n", threadIdx.x, d_sum[0]);
//	__threadfence();
}



typedef float* FP;

int GPU_Stream_test(float *h_input, float *h_output, size_t nElements, int nStreams){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount){
		checkCudaErrors(cudaSetDevice(device));
	}
	else {
		printf("ERROR! Selected device is not available\n");
		return(1);
	}
	cudaMemGetInfo(&free_mem,&total_mem);
	
	//---------> Checking memory
	int nElements_per_stream = nElements/nStreams;
	size_t input_size  = nElements*sizeof(float);
	size_t input_size_per_stream  = nElements_per_stream*sizeof(float);
	size_t output_size = nStreams*sizeof(float);
	size_t output_size_per_stream = nStreams*sizeof(float);
	if((input_size+output_size)>free_mem){
		printf("Not enough memory!\n");
		return(1);
	}
	
	/*
	//--------------> No Streams
	float *d_input;
	float *d_output;
	checkCudaErrors(cudaMalloc(&d_input,  input_size) );
	checkCudaErrors(cudaMalloc(&d_output, output_size ) );
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_output, 0, output_size));
	
	//-----------------------------> Kernel
	dim3 gridSize(nElements_per_stream/128, 1 , 1);
	dim3 blockSize(128, 1, 1);
	for(int f = 0; f < nStreams; f++){
		reduce_using_atomics<<<gridSize,blockSize>>>(&d_input[f*nElements_per_stream], &d_output[f], nElements_per_stream);
	}
		
	checkCudaErrors(cudaMemcpy( h_output, d_output, output_size, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	//---------------------------------------------------------<
	*/
	
	//---------------------> With as many streams as needed, that is we have enough memory to store whole task and we just chop it into stream chunks
	omp_set_num_threads(nStreams);
	//omp_set_num_threads(1);
	
//	cudaStream_t *streams;
//	streams = new cudaStream_t[nStreams];
	
	#pragma omp parallel shared(h_input, h_output)
	{
		int th_id = omp_get_thread_num();
		int nthreads = omp_get_num_threads();
		cudaStream_t local_stream;
		cudaStreamCreate(&local_stream);
		printf("Thread-stream: %d;\n", th_id);
		
		float *d_input_ps;
		float *d_output_ps;
		unsigned int *d_local_count;
		float *d_local_sum;
		checkCudaErrors( cudaMalloc(&d_input_ps,  input_size_per_stream) );
		checkCudaErrors( cudaMalloc(&d_output_ps, output_size_per_stream ) );
		checkCudaErrors( cudaMalloc(&d_local_count, sizeof(unsigned int) ) );
		checkCudaErrors( cudaMalloc(&d_local_sum, sizeof(float) ) );
		
		checkCudaErrors( cudaMemset(d_local_count, 0, sizeof(unsigned int) ) );
		checkCudaErrors( cudaMemset(d_local_sum, 0, sizeof(float) ) );
		checkCudaErrors( cudaMemcpyAsync( d_input_ps, &h_input[th_id*nElements_per_stream], input_size_per_stream, cudaMemcpyHostToDevice, local_stream) );
		
		dim3 gridSize(nElements_per_stream/128, 1 , 1);
		dim3 blockSize(128, 1, 1);
		reduce_using_atomics<<< gridSize , blockSize , 0 , local_stream >>>(d_output_ps, d_input_ps, d_local_count, d_local_sum, nElements_per_stream);
		cudaStreamSynchronize(local_stream);

//		dummy_copy<<<1,1,0,local_stream>>>(d_output_ps, d_local_sum);
//		cudaStreamSynchronize(local_stream);
		
		while(cudaStreamQuery(local_stream)!=cudaSuccess){
			printf("Waiting %d!", th_id);
		}
		printf("Finished! %d;\n", th_id);
		
		checkCudaErrors( cudaMemcpyAsync( &h_output[th_id], d_output_ps, output_size_per_stream, cudaMemcpyDeviceToHost, local_stream) );
//		cudaStreamAttachMemAsync ( local_stream, &h_output[th_id], output_size_per_stream, cudaMemAttachSingle );
//		cudaStreamSynchronize(local_stream);
		
		while(cudaStreamQuery(local_stream)!=cudaSuccess){
			printf("Waiting %d!", th_id);
		}
		printf("Finished! %d;\n", th_id);
		
		checkCudaErrors(cudaFree(d_input_ps));
		checkCudaErrors(cudaFree(d_output_ps));
		checkCudaErrors(cudaFree(d_local_count));
		checkCudaErrors(cudaFree(d_local_sum));
		
		cudaStreamDestroy(local_stream);
	}
	
	
	return(0);
}


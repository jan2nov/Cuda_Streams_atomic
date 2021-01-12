#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 

#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;

#define VERBOSE true

//-------------------------------------------------------------------------------
//-----------> Data manipulations
void Export_to_file(float *data, int nRows, int nColumns, int offset, char *filename){
	int r, c;
	
	ofstream FILEOUT;
	FILEOUT.open(filename);
	for(r=0; r<nRows; r++){
		for(c=0; c<nColumns-offset; c++){
			FILEOUT << r << " " << c << " " << data[r*nColumns+c] << endl;
		}
		FILEOUT << endl;
	}
	FILEOUT.close();
}

void Compare_data(float *CPU_output, float *GPU_output, int nStreams){
	double merror,terror;
	
	terror = 0;
	for(int f = 0; f < nStreams; f++){
		double difference = CPU_output[f] - GPU_output[f];
		terror = terror + difference;
		printf("%f - %f = %e\n", CPU_output[f], GPU_output[f], difference);
	}
	
	merror = terror/nStreams;
	printf("Mean error = %e;\n", merror);
}

// simplest even thought is not the most precise method
void Reduction(float *h_input, float *h_output, int nStreams, size_t nElements_per_stream){
	for(int st_id = 0; st_id < nStreams; st_id++){
		double sum = 0;
		for(size_t f = 0; f < nElements_per_stream; f++){
			size_t pos = st_id*nElements_per_stream + f;
			sum = sum + h_input[pos];
		}
		h_output[st_id] = sum;
	}
}
//-----------> Data manipulations
//-------------------------------------------------------------------------------



//-------------------------------------------------------------------------------
//-----------> Generating data
void Generate_dataset(float *h_input, size_t nElements){
	for(size_t f=0; f<nElements; f++){
		h_input[f] = rand() / (float)RAND_MAX;
	}
}
//-----------> Generating data
//-------------------------------------------------------------------------------


int GPU_Stream_test(float *h_input, float *h_output, size_t nElements, int nStreams);

int main(int argc, char* argv[]) {
	//----------------> Program parameters
	int nStreams;

	// Check!
	char * pEnd;
	if (argc==2) {
		nStreams = strtol(argv[1],&pEnd,10);
	}
	else {
		printf("Argument error!\n");
		printf(" 1) number of Streams\n");
        return 1;
	}
	
	printf("Arguments\n");
	printf("nStreams:%d\n",nStreams);
	//----------------> Program parameters
	
	//----------------> Task related 
	
	//----------------> Other 
	int nThreads_per_block = 128;
	size_t nElements_per_stream = 327680;
	size_t nElements = (size_t)(nStreams)*nElements_per_stream;
	
	printf("Number of threads per CUDAblock: %d;\n", nThreads_per_block);
	printf("Elements per stream: %zu;\n", nElements_per_stream);
	printf("Total number of elements: %zu;\n", nElements);	
	

	float *h_input;
	float *h_GPU_output;
	float *h_CPU_output;

	if (VERBOSE) printf("Host memory allocation...\t");
		h_input		  = (float *)malloc(nElements*sizeof(float));
		h_GPU_output	  = (float *)malloc(nStreams*sizeof(float));
		h_CPU_output	  = (float *)malloc(nStreams*sizeof(float));
	if (VERBOSE) printf("done.\n");

	if (VERBOSE) printf("Host memory memset...\t\t");
		memset(h_input, 0.0, nElements*sizeof(float));
		memset(h_GPU_output, 1, nStreams*sizeof(float));
		memset(h_CPU_output, 0, nStreams*sizeof(float));
	if (VERBOSE) printf("done.\n");
	fflush(stdout);

	if (VERBOSE) printf("Creating random data set...\t");
		srand(time(NULL));
		Generate_dataset(h_input, nElements);
	if (VERBOSE) printf("done.\n");
	
	GPU_Stream_test(h_input, h_GPU_output, nElements, nStreams);
	
	Reduction(h_input, h_CPU_output, nStreams, nElements_per_stream);
	Compare_data(h_CPU_output, h_GPU_output, nStreams);
	
	free(h_input);
	free(h_GPU_output);
	free(h_CPU_output);
	
	cudaDeviceReset();
	
	if (VERBOSE) printf("Finished!\n");

	return (0);
}

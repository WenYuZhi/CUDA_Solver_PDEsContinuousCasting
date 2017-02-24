
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "book.h"
#include "gridcheck.h"

# define Section 12  // number of cooling sections

cudaError_t addWithCuda(float *T_Init, float dx, float dy, float tao, int nx, int ny, int tnpts, float *, float *, int num_blocks, int num_threadsx, int num_threadsy);
__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd);
__device__ float Boundary_Condition(int j, float dx, float *ccml_zone, float *H_Init);

__global__ void MainKernel(float *T_New, float *T_Last, float *ccml, float *H_Init, float dx, float dy, float tao, int nx, int ny, bool disout)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = threadIdx.y;
	int idx = j * ny + i;
	int D = ny;

	float pho, Ce, lamd; // physical parameters pho represents desity, Ce is specific heat and lamd is thermal conductivity
	float a, T_Up, T_Down, T_Right, T_Left, T_Middle, h = 0.0, Tw = 30.0, Vcast = -0.02, T_Cast = 1558.0;



	if (disout) {
		Physicial_Parameters(T_Last[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho*Ce);
		h = Boundary_Condition(i, dy, ccml, H_Init);
		if (j == 0 && i != 0 && i != ny - 1) //1
		{
			T_Middle = T_Last[idx];
			T_Up = T_Last[idx + D];
			T_Down = T_Last[idx + D] - 2 * dx * h * (T_Middle - Tw) / lamd;
			T_Right = T_Last[idx + 1];
			T_Left = T_Last[idx - 1];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j == nx - 1 && i != 0 && i != ny - 1)  //2
		{
			T_Middle = T_Last[idx];
			T_Up = T_Last[idx - D];
			T_Down = T_Last[idx - D];
			T_Right = T_Last[idx + 1];
			T_Left = T_Last[idx - 1];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (i == 0)  //3
		{
			T_New[idx] = T_Cast;
		}
		else if (i == ny - 1 && j != 0 && j != nx - 1)  //4
		{
			T_Middle = T_Last[idx];
			T_Up = T_Last[idx + D];
			T_Down = T_Last[idx - D];
			T_Right = T_Last[idx - 1];
			T_Left = T_Last[idx - 1];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j == 0 && i == ny - 1)  //5
		{
			T_Middle = T_Last[idx];
			T_Up = T_Last[idx + D];
			T_Down = T_Last[idx + D];
			T_Right = T_Last[idx - 1];
			T_Left = T_Last[idx - 1];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j == nx - 1 && i == ny - 1) //6
		{
			T_Middle = T_Last[idx];
			T_Up = T_Last[idx - D];
			T_Down = T_Last[idx - D];
			T_Right = T_Last[idx - 1];
			T_Left = T_Last[idx - 1];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j != 0 && j != nx - 1 && i != 0 && i != ny - 1) //7
		{
			T_Middle = T_Last[idx];
			T_Up = T_Last[idx + D];
			T_Down = T_Last[idx - D];
			T_Right = T_Last[idx + 1];
			T_Left = T_Last[idx - 1];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
	}

	else
	{
		Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho*Ce);
		h = Boundary_Condition(i, dy, ccml, H_Init);
		if (j == 0 && i != 0 && i != ny - 1) //1
		{
			T_Middle = T_New[idx];
			T_Up = T_New[idx + D];
			T_Down = T_New[idx + D] - 2 * dx * h* (T_Middle - Tw) / lamd;
			T_Right = T_New[idx + 1];
			T_Left = T_New[idx - 1];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j == nx - 1 && i != 0 && i != ny - 1)  //2
		{
			T_Middle = T_New[idx];
			T_Up = T_New[idx - D];
			T_Down = T_New[idx - D];
			T_Right = T_New[idx + 1];
			T_Left = T_New[idx - 1];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (i == 0)  //3
		{
			T_Last[idx] = T_Cast;
		}
		else if (i == ny - 1 && j != 0 && j != nx - 1)  //4
		{
			T_Middle = T_New[idx];
			T_Up = T_New[idx + D];
			T_Down = T_New[idx - D];
			T_Right = T_New[idx - 1];
			T_Left = T_New[idx - 1];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j == 0 && i == ny - 1)  //5
		{
			T_Middle = T_New[idx];
			T_Up = T_New[idx + D];
			T_Down = T_New[idx + D];
			T_Right = T_New[idx - 1];
			T_Left = T_New[idx - 1];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j == nx - 1 && i == ny - 1) //6
		{
			T_Middle = T_New[idx];
			T_Up = T_New[idx - D];
			T_Down = T_New[idx - D];
			T_Right = T_New[idx - 1];
			T_Left = T_New[idx - 1];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
		else if (j != 0 && j != nx - 1 && i != 0 && i != ny - 1)  //7
		{
			T_Middle = T_New[idx];
			T_Up = T_New[idx + D];
			T_Down = T_New[idx - D];
			T_Right = T_New[idx + 1];
			T_Left = T_New[idx - 1];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Middle +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
		}
	}
}

int main()
{
	const int nx = 11, ny = 3000;   // nx is the number of grid in x direction, ny is the number of grid in y direction.
	int num_blocks = 1, num_threadsx = 1, num_threadsy = 1; // block number(1D)  thread number in x and y dimension(2D)
	int tnpts = 10000;  // time step
	float T_Cast = 1558.0, Lx = 0.125, Ly = 28.599, t_final = 2000.0, dx, dy, tao;  // T_Cast is the casting temperature Lx and Ly is the thick and length of steel billets
	float *T_Init;
	float ccml[Section + 1] = { 0.0,0.2,0.4,0.6,0.8,1.0925,2.27,4.29,5.831,9.6065,13.6090,19.87014,28.599 }; // The cooling sections
	float H_Init[Section] = { 1380,1170,980,800,1223.16,735.05,424.32,392.83,328.94,281.64,246.16,160.96 };  // The heat transfer coefficients in the cooling sections

	T_Init = (float *)calloc(nx * ny, sizeof(float));  // Initial condition

	num_threadsy = nx;
	num_threadsx = 30;
	num_blocks = ny / num_threadsx;
	if (ny % num_threadsx != 0)  // ny mod num_threadsx must be 0
	{
		printf("The number of threadsx is error. Please check the variable \"num_threadsx\"");
		exit(0);
	}
	
	for (int i = 0; i < nx; i++)
		for (int j = 0; j < ny; j++)
			T_Init[i * ny + j] = T_Cast;  // give the initial condition

	dx = Lx / (nx - 1);            // the grid size x
	dy = Ly / (ny - 1);            // the grid size y
	tao = t_final / (tnpts - 1);   // the time step size
	gridcheck(dx, dy, tao);

	printf("Casting Temperature = %f ", T_Cast);
	printf("\n");
	printf("The thick of steel billets(m) = %f ", Lx);
	printf("\n");
	printf("The length of steel billets(m) = %f ", Ly);
	printf("\n");
	printf("dx(m) = %f ", dx);
	printf("dy(m) = %f ", dy);
	printf("tao(s) = %f ", tao);
	printf("\n");
	printf("simulation time(s) = %f\n ", t_final);

	clock_t timestart = clock();
	cudaError_t cudaStatus = addWithCuda(T_Init, dx, dy, tao, nx, ny, tnpts, ccml, H_Init, num_blocks, num_threadsx, num_threadsy);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	clock_t timeend = clock();

	printf("running time = %d(millisecond)", (timeend - timestart));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t addWithCuda(float *T_Init, float dx, float dy, float tao, int nx, int ny, int tnpts, float *ccml, float *H_Init, int num_blocks, int num_threadsx, int num_threadsy)
{
	float *dev_T_New, *dev_T_Last, *dev_ccml, *dev_H_Init; // the point on GPU
	float *T_Result;
	const int Num_Iter = 500;                         // The result can be obtained by every Num_Iter time step
	volatile bool dstOut = true;
	FILE *fp = NULL;

	T_Result = (float *)calloc(nx * ny, sizeof(float)); // The temperature of steel billets

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New, nx * ny * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last, nx * ny * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_ccml, (Section + 1) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init, Section * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_T_Last, T_Init, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_ccml, ccml, (Section + 1) * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_H_Init, H_Init, Section * sizeof(float), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(num_threadsx, num_threadsy);
	dim3 numBlocks(num_blocks);
	// Launch a kernel on the GPU with one thread for each element.
	for (int i = 0; i < tnpts; i++)
	{
		MainKernel << <numBlocks, threadsPerBlock >> >(dev_T_New, dev_T_Last, dev_ccml, dev_H_Init, dx, dy, tao, nx, ny, dstOut);
		dstOut = !dstOut;

		if (i % Num_Iter == 0) {
			HANDLE_ERROR(cudaMemcpy(T_Result, dev_T_Last, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
			printf("time_step = %d  simulation time is %f\n", i, i*tao);
			printf("%f, %f, %f", T_Result[0], T_Result[(nx - 1)*(ny - 1) - nx], T_Result[(nx - 1)*(ny - 1)]);
			printf("\n");
		}
	}

	fp = fopen("C:\\Temperature2DGPU_Static.txt", "w");
	for (int i = 0; i < nx; i++)
	{
	    for (int j = 0; j < ny; j++)
	        fprintf(fp, " %f", T_Result[i * ny + j]);
	    fprintf(fp, "\n");
	}
	fclose(fp);

	// Check for any errors launching the kernel
	HANDLE_ERROR(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.


Error:
	cudaFree(dev_T_New);
	cudaFree(dev_T_Last);
	cudaFree(dev_ccml);
	cudaFree(dev_H_Init);

	return cudaStatus;
}
// Helper function for using CUDA to add vectors in parallel.

__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd)
{
	float Ts = 1462.0, Tl = 1518.0, lamds = 30, lamdl = 50, phos = 7000, phol = 7500, ce = 540.0, L = 265600.0, fs = 0.0;
	if (T<Ts)
	{
		fs = 0;
		*pho = phos;
		*lamd = lamds;
		*Ce = ce;
	}

	if (T >= Ts&&T <= Tl)
	{
		fs = (T - Ts) / (Tl - Ts);
		*pho = fs*phos + (1 - fs)*phol;
		*lamd = fs*lamds + (1 - fs)*lamdl;
		*Ce = ce + L / (Tl - Ts);
	}

	if (T>Tl)
	{
		fs = 1;
		*pho = phol;
		*lamd = lamdl;
		*Ce = ce;
	}

}

__device__ float Boundary_Condition(int j, float dy, float *ccml_zone, float *H_Init)
{
	float YLabel, h = 0.0;
	YLabel = j*dy;

	for (int i = 0; i < Section; i++)
	{
		if (YLabel >= *(ccml_zone + i) && YLabel <= *(ccml_zone + i + 1))
			h = *(H_Init + i);
	}
	return h;
}
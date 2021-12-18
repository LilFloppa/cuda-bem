#include "cuda_runtime.h"

#include "Point.h"

__host__ __device__ float F1(Vector x, Vector y, float p)
{
	return p / (4 * 3.14159265358979323846f * (x - y).Norm());
}

__host__ __device__ float F2(Vector x, Vector y, Vector n, float q)
{
	float norm = (x - y).Norm();
	return ((n * q) * (x - y)) / (4 * 3.14159265358979323846f * norm * norm * norm);
}

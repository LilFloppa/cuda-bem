#include "cuda_runtime.h"

#include "Point.h"

__host__ __device__ double F1(Vector x, Vector y, double p)
{
	return p / (4 * 3.14159265358979323846 * (x - y).Norm());
}

__host__ __device__ double F2(Vector x, Vector y, Vector n, double q)
{
	double norm = (x - y).Norm();
	return ((n * q) * (x - y)) / (4 * 3.14159265358979323846 * norm * norm * norm);
}

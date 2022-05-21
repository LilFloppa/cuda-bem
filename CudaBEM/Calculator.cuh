#pragma once
#include "Point.h"
#include "Element.h"
#include "Gauss.cuh"
#include "Funcs.cuh"

#include <vector>
using namespace std;

struct calculator
{
	size_t node_count;
	Point* points;
	Element* elements;
	double* q;
	double* p;
};

void create_calculator(vector<Point>& points, vector<Element>& elements, vector<double>& q, vector<double>& p, calculator** bem_calculator)
{
	calculator* calc = new calculator;
	calc->node_count = elements.size();

	cudaMalloc((void**)&(calc->points), points.size() * sizeof(Point));
	cudaMalloc((void**)&(calc->elements), elements.size() * sizeof(Element));
	cudaMalloc((void**)&(calc->q), q.size() * sizeof(double));
	cudaMalloc((void**)&(calc->p), p.size() * sizeof(double));

	cudaMemcpy(calc->points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(calc->elements, elements.data(), elements.size() * sizeof(Element), cudaMemcpyHostToDevice);
	cudaMemcpy(calc->q, q.data(), q.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(calc->p, p.data(), p.size() * sizeof(double), cudaMemcpyHostToDevice);

	*bem_calculator = calc;
}

void dispose_calculator(calculator* calc)
{
	cudaFree(calc->elements);
	cudaFree(calc->points);
	cudaFree(calc->p);
	cudaFree(calc->q);
}

__global__ void integrate_kernel(calculator* bem_calc, double x, double y, double z, double* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= bem_calc->node_count)
		return;

	Element* elements = bem_calc->elements;
	Point* points = bem_calc->points;
	double* q = bem_calc->q;
	double* p = bem_calc->p;

	Element el = elements[index];
	Point A = points[el.v1];
	Point B = points[el.v2];
	Point C = points[el.v3];

	Vector v1 = B - A;
	Vector v2 = C - A;
	Vector temp = v1.Cross(v2).Normalize();
	double3 normal = make_double3(temp.x, temp.y, temp.z);

	double3 Q = make_double3(q[el.q1], q[el.q2], q[el.q3]);
	double DuDn = p[el.p];

	double3 X;
	double3 Y = make_double3(x, y, z);

	double3 xx = make_double3(A.x, B.x, C.x);
	double3 yy = make_double3(A.y, B.y, C.y);
	double3 zz = make_double3(A.z, B.z, C.z);

	for (int i = 0; i < 66; i++)
	{
		double ksi = p1[i];
		double etta = p2[i];
		double weight = w[i];

		double3 L = make_double3(1 - ksi - etta, ksi, etta);

		double U = L.x * Q.x + L.y * Q.y + L.z * Q.z;
		X.x = L.x * xx.x + L.y * xx.y + L.z * xx.z;
		X.y = L.x * yy.x + L.y * yy.y + L.z * yy.z;
		X.z = L.x * zz.x + L.y * zz.y + L.z * zz.z;

		double3 diff = make_double3(X.x - Y.x, X.y - Y.y, X.z - Y.z);
		double norm = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
		double coeff = U * (normal.x * diff.x + normal.y * diff.y + normal.z * diff.z);

		double f = DuDn / (4 * 3.14159265358979323846 * norm);
		f += (coeff) / (4 * 3.14159265358979323846 * norm * norm * norm);

		result[index] += 0.25 * weight * f;
	}

	result[index] *= v1.Cross(v2).Norm();
}

double calculate_value(calculator* bem_calc, double x, double y, double z)
{
	size_t size = bem_calc->node_count;

	double* result;
	cudaMalloc((void**)&result, size * sizeof(double));
	cudaMemset(result, 0, size * sizeof(double));

	calculator* dev_calc;
	cudaMalloc(&dev_calc, sizeof(calculator));
	cudaMemcpy(dev_calc, bem_calc, sizeof(calculator), cudaMemcpyHostToDevice);

	size_t block_size = 256;
	size_t block_count = 1;
	while (block_count * block_size < size)
		block_count++;

	integrate_kernel<<<block_count, block_size>>>(dev_calc, x, y, z, result);
	cudaDeviceSynchronize();

	double res = 0.0;

	double* host_result = new double[size];
	cudaMemcpy(host_result, result, size * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
		res += host_result[i];

	return res;
}
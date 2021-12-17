#pragma once
#include "Point.h"
#include "Element.h"
#include "Gauss.cuh"
#include "Funcs.cuh"

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

__global__ void integrate_kernel(calculator* bem_calc, double x, double y, double z, double* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	size_t node_count = bem_calc->node_count;
	Element* elements = bem_calc->elements;
	Point* points = bem_calc->points;
	double* q = bem_calc->q;
	double* p = bem_calc->p;

	if (index >= node_count)
		return;

	Element el = elements[index];
	Point A = points[el.v1];
	Point B = points[el.v2];
	Point C = points[el.v3];

	Vector v1 = B - A;
	Vector v2 = C - A;
	Vector normal = v1.Cross(v2).Normalize();

	Vector Q(q[el.q1], q[el.q2], q[el.q3]);
	double DuDn = p[el.p];

	Vector X;
	Point Y(x, y, z);

	for (int i = 0; i < 66; i++)
	{
		double ksi = p1[i];
		double etta = p2[i];
		double weight = w[i];

		Point L(1 - ksi - etta, ksi, etta);

		double U = L * Q;
		X.x = L * Vector(A.x, B.x, C.x);
		X.y = L * Vector(A.y, B.y, C.y);
		X.z = L * Vector(A.z, B.z, C.z);

		double f = F1(X, Y, DuDn) + F2(X, Y, normal, U);

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

	integrate_kernel << <1, 256 >> > (dev_calc, x, y, z, result);
	cudaDeviceSynchronize();

	double res = 0.0;

	double* host_result = new double[size];
	cudaMemcpy(host_result, result, size * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
		res += host_result[i];

	return res;
}
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
	float* q;
	float* p;
};

void create_calculator(vector<Point>& points, vector<Element>& elements, vector<float>& q, vector<float>& p, calculator** bem_calculator)
{
	calculator* calc = new calculator;
	calc->node_count = elements.size();

	cudaMalloc((void**)&(calc->points), points.size() * sizeof(Point));
	cudaMalloc((void**)&(calc->elements), elements.size() * sizeof(Element));
	cudaMalloc((void**)&(calc->q), q.size() * sizeof(float));
	cudaMalloc((void**)&(calc->p), p.size() * sizeof(float));

	cudaMemcpy(calc->points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(calc->elements, elements.data(), elements.size() * sizeof(Element), cudaMemcpyHostToDevice);
	cudaMemcpy(calc->q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(calc->p, p.data(), p.size() * sizeof(float), cudaMemcpyHostToDevice);

	*bem_calculator = calc;
}

__global__ void integrate_kernel(calculator* bem_calc, float x, float y, float z, float* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= bem_calc->node_count)
		return;

	Element* elements = bem_calc->elements;
	Point* points = bem_calc->points;
	float* q = bem_calc->q;
	float* p = bem_calc->p;

	Element el = elements[index];
	Point A = points[el.v1];
	Point B = points[el.v2];
	Point C = points[el.v3];

	Vector v1 = B - A;
	Vector v2 = C - A;
	Vector temp = v1.Cross(v2).Normalize();
	float3 normal = make_float3(temp.x, temp.y, temp.z);

	float3 Q = make_float3(q[el.q1], q[el.q2], q[el.q3]);
	float DuDn = p[el.p];

	float3 X;
	float3 Y = make_float3(x, y, z);

	float3 xx = make_float3(A.x, B.x, C.x);
	float3 yy = make_float3(A.y, B.y, C.y);
	float3 zz = make_float3(A.z, B.z, C.z);

	for (int i = 0; i < 66; i++)
	{
		float ksi = p1[i];
		float etta = p2[i];
		float weight = w[i];

		float3 L = make_float3(1 - ksi - etta, ksi, etta);

		float U = L.x * Q.x + L.y * Q.y + L.z * Q.z;
		X.x = L.x * xx.x + L.y * xx.y + L.z * xx.z;
		X.y = L.x * yy.x + L.y * yy.y + L.z * yy.z;
		X.z = L.x * zz.x + L.y * zz.y + L.z * zz.z;

		float3 diff = make_float3(X.x - Y.x, X.y - Y.y, X.z - Y.z);
		float norm = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
		float coeff = U * (normal.x * diff.x + normal.y * diff.y + normal.z * diff.z);

		float f = DuDn / (4 * 3.14159265358979323846 * norm);
		f += (coeff) / (4 * 3.14159265358979323846 * norm * norm * norm);

		result[index] += 0.25 * weight * f;
	}

	result[index] *= v1.Cross(v2).Norm();
}

float calculate_value(calculator* bem_calc, float x, float y, float z)
{
	size_t size = bem_calc->node_count;

	float* result;
	cudaMalloc((void**)&result, size * sizeof(float));
	cudaMemset(result, 0, size * sizeof(float));

	calculator* dev_calc;
	cudaMalloc(&dev_calc, sizeof(calculator));
	cudaMemcpy(dev_calc, bem_calc, sizeof(calculator), cudaMemcpyHostToDevice);

	size_t block_size = 256;
	size_t block_count = (size + block_size - 1) / size;

	integrate_kernel<<<block_count, block_size>>>(dev_calc, x, y, z, result);
	cudaDeviceSynchronize();

	float res = 0.0;

	float* host_result = new float[size];
	cudaMemcpy(host_result, result, size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
		res += host_result[i];

	return res;
}
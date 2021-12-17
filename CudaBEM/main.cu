#include <iostream>
#include <chrono>
#include <vector>

#include "cuda_runtime.h"
#include "Point.h"
#include "Element.h"
#include "Input.h"
#include "Gauss.cuh"
#include "Calculator.cuh"


void integrate_index(size_t index, size_t element_count, Point* points, Element* elements, double* q, double* p, Vector Y, double* result)
{
	if (index >= element_count)
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

	for (int i = 0; i < 66; i++)
	{
		double ksi = p1h[i];
		double etta = p2h[i];
		double weight = wh[i];

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

double Integrate(vector<Point>& points, vector<Element>& elements, vector<double>& q, vector<double>& p, Vector& Y)
{
	double result = 0;

	double* result_array = new double[elements.size()];
	int index = 0;
	for (auto& el : elements)
	{
		Point A = points[el.v1];
		Point B = points[el.v2];
		Point C = points[el.v3];

		Vector v1 = B - A;
		Vector v2 = C - A;

		Vector normal = v1.Cross(v2).Normalize();

		Vector Q(q[el.q1], q[el.q2], q[el.q3]);
		double DuDn = p[el.p];

		Vector X;

		double res_i = 0;
		for (int i = 0; i < 66; i++)
		{
			double ksi = p1h[i];
			double etta = p2h[i];
			double weight = wh[i];

			Point L(1 - ksi - etta, ksi, etta);

			double U = L * Q;
			X.x = L * Vector(A.x, B.x, C.x);
			X.y = L * Vector(A.y, B.y, C.y);
			X.z = L * Vector(A.z, B.z, C.z);

			double f = F1(X, Y, DuDn) + F2(X, Y, normal, U);

			res_i += 0.25 * weight * f;
		}

		res_i *= v1.Cross(v2).Norm();

		result_array[index++] = res_i;
	}


	for (int i = 0; i < elements.size(); i++)
		result += result_array[i];

	return result;
}

double IntegrateIndex(vector<Point>& points, vector<Element>& elements, vector<double>& q, vector<double>& p, Vector& Y)
{
	size_t size = elements.size();

	double* host_result = new double[size];
	memset(host_result, 0, size * sizeof(double));

	for (int i = 0; i < size; i++)
		integrate_index(i, elements.size(), points.data(), elements.data(), q.data(), p.data(), Y, host_result);

	double res = 0.0;
	for (int i = 0; i < size; i++)
		res += host_result[i];

	return res;
}

int main(void)
{
	std::vector<Point> points;
	std::vector<double> q;
	std::vector<double> p;
	std::vector<Element> elements;

	Input("test/points.txt", points, "test/weights.txt", q, p, "test/triangles.txt", elements);

	Point target;

	ifstream in("test/target.txt");
	in >> target.x >> target.y >> target.z;
	in.close();

	auto start = std::chrono::steady_clock::now();
	double result = Integrate(points, elements, q, p, target);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Integrate. Elapsed time: " << elapsed_seconds.count() << "s\n";

	calculator* calc = nullptr;
	create_calculator(points, elements, q, p, &calc);

	start = std::chrono::steady_clock::now();
	double resultCuda = calculate_value(calc, target.x, target.y, target.z);
	end = std::chrono::steady_clock::now();
	elapsed_seconds = end - start;
	std::cout << "IntegrateCuda. Elapsed time: " << elapsed_seconds.count() << "s\n";

	start = std::chrono::steady_clock::now();
	double resultIndex = IntegrateIndex(points, elements, q, p, target);
	end = std::chrono::steady_clock::now();
	elapsed_seconds = end - start;
	std::cout << "IntegrateIndex. Elapsed time: " << elapsed_seconds.count() << "s\n";

	return 0;
}
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>
#include <iomanip>

#include <omp.h>

#include "Point.h"
#include "Element.h"
#include "Input.h"
#include "Gauss.h"

using namespace std;

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
			double ksi = p1[i];
			double etta = p2[i];
			double weight = w[i];

			Point L(1 - ksi - etta, ksi, etta);

			double U = L * Q;
			X.x = L * Vector(A.x, B.x, C.x);
			X.y = L * Vector(A.y, B.y, C.y);
			X.z = L * Vector(A.z, B.z, C.z);

			double norm = (X - Y).Norm();
			double f = DuDn / (4 * 3.14159265358979323846 * norm);
			f += ((normal * U) * (X - Y)) / (4 * 3.14159265358979323846 * norm * norm * norm);

			res_i += 0.25 * weight * f;
		}

		res_i *= v1.Cross(v2).Norm();

		result += res_i;
	}

	return result;
}

void Cycle(std::ofstream& out, int i)
{
	vector<Point> points;
	vector<double> q;
	vector<double> p;
	vector<Element> elements;

	Input("test4/points.txt", points, "test4/weights.txt", q, p, "test4/triangles.txt", elements, i);

	Point target;

	ifstream in("test4/target.txt");
	in >> target.x >> target.y >> target.z;
	in.close();

	auto start = std::chrono::steady_clock::now();
	double result = Integrate(points, elements, q, p, target);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	std::cout << "Element Count: " << elements.size() << " Elapsed time: " << elapsed_seconds.count() * 1000.0 << "ms\n";
	out << elapsed_seconds.count() * 1000.0 << std::endl;
}

int main()
{
	std::ofstream out("result.txt");


		Cycle(out, 1000);

	out.close();
	return 0;
}
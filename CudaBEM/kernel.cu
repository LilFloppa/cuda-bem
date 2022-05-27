#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <iomanip>

#include "cuda_runtime.h"
#include "Point.h"
#include "Element.h"
#include "Input.h"
#include "Gauss.cuh"
#include "Calculator.cuh"
#include "Integrator.h"

struct cycle_data
{
	size_t el_count;
	double result;
	double time;
};

cycle_data Cycle(int i)
{
	vector<Point> points;
	vector<double> q;
	vector<double> p;
	vector<Element> elements;

	Input("test/points.txt", points, "test/weights.txt", q, p, "test/triangles.txt", elements, i);

	Point target;

	ifstream in("test/target.txt");
	in >> target.x >> target.y >> target.z;
	in.close();

	auto start = std::chrono::steady_clock::now();
	double result = Integrate(points, elements, q, p, target);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	return cycle_data{ elements.size(), result, elapsed_seconds.count() };
}

cycle_data CycleCuda(int i)
{
	std::vector<Point> points;
	std::vector<double> q;
	std::vector<double> p;
	std::vector<Element> elements;

	Input("test/points.txt", points, "test/weights.txt", q, p, "test/triangles.txt", elements, i);

	Point target;

	ifstream in("test/target.txt");
	in >> target.x >> target.y >> target.z;
	in.close();

	auto start = std::chrono::steady_clock::now();
	calculator* calc = nullptr;
	create_calculator(points, elements, q, p, &calc);

	double resultCuda = calculate_value(calc, 0, nullptr, target.x, target.y, target.z);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	dispose_calculator(calc);

	return cycle_data{ elements.size(), resultCuda, elapsed_seconds.count() };
}

int main(void)
{
	vector<Point> points;
	vector<double> q;
	vector<double> p;
	vector<Element> elements;

	Input("test/points.txt", points, "test/weights.txt", q, p, "test/triangles.txt", elements, 100);

	Point target(0.3, 0.25, 0.6);

	int n = 50000;
	auto start = std::chrono::steady_clock::now();
	double sum = 0.0;
	for (int i = 0; i < n; i++)
	{
		double result = Integrate(points, elements, q, p, target);
		sum += result;
	}
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "CPU: " << elapsed_seconds.count() << " SUM: " << std::setprecision(15) << sum << std::endl;

	double* host_result = new double[elements.size()];
	calculator* calc = nullptr;
	create_calculator(points, elements, q, p, &calc);
	auto cudaStart = std::chrono::steady_clock::now();
	double cudaSum = 0.0;
	for (int i = 0; i < n; i++)
	{
		double resultCuda = calculate_value(calc, elements.size(), host_result, target.x, target.y, target.z);
		cudaSum += resultCuda;
	}
	auto cudaEnd = std::chrono::steady_clock::now();
	std::chrono::duration<double> cuda_elapsed_seconds = cudaEnd - cudaStart;
	std::cout << "GPU CUDA: " << cuda_elapsed_seconds.count() << std::setprecision(15) << " SUM: " << cudaSum << std::endl;
	dispose_calculator(calc);

	return 0;
}
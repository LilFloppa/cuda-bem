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

	return cycle_data{ elements.size(), result, elapsed_seconds.count() * 1000.0 };
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

	double resultCuda = calculate_value(calc, target.x, target.y, target.z);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	dispose_calculator(calc);

	return cycle_data{ elements.size(), resultCuda, elapsed_seconds.count() * 1000.0 };
}

int main(void)
{
	std::ofstream out("result.txt");
	std::ofstream outCuda("resultCuda.txt");

	int count = 100;
	int mult = 200;
	std::vector<cycle_data> times(mult);
	std::vector<cycle_data> timesCuda(mult);
	for (int k = 0; k < count; k++)
	{
		for (int i = 1; i <= mult; i++)
		{
			auto pair = Cycle(i);
			times[i - 1].el_count = pair.el_count;
			times[i - 1].result = pair.result;
			times[i - 1].time += pair.time;

			auto pairCuda = CycleCuda(i);
			timesCuda[i - 1].el_count = pairCuda.el_count;
			timesCuda[i - 1].result = pairCuda.result;
			timesCuda[i - 1].time += pairCuda.time;
		}

		std::cout << "Iteration: " << k << " completed" << std::endl;
	}

	for (int i = 0; i < mult; i++)
	{
		times[i].time /= count;
		timesCuda[i].time /= count;
	}

	for (int i = 0; i < mult; i++)
	{
		out << times[i].el_count << " " << times[i].time << " " << times[i].result << std::endl;
		outCuda << timesCuda[i].el_count << " " << timesCuda[i].time << " " << timesCuda[i].result << std::endl;
	}


	out.close();
	outCuda.close();
	return 0;
}
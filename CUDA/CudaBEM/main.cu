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

struct my_pair
{
	size_t el_count;
	double time;
};

my_pair Cycle(std::ofstream& out, int i)
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


	calculator* calc = nullptr;
	create_calculator(points, elements, q, p, &calc);
	auto start = std::chrono::steady_clock::now();
	double resultCuda = calculate_value(calc, target.x, target.y, target.z);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	dispose_calculator(calc);

	//std::cout << "Element Count: " << elements.size() << " Elapsed time: " << elapsed_seconds.count() * 1000.0 << "ms\n";
	//out << elements.size() << " " << elapsed_seconds.count() * 1000.0 << std::endl;

	return my_pair{ elements.size(), elapsed_seconds.count() * 1000.0 };
}

int main(void)
{
	std::ofstream out("result.txt");

	int count = 100;
	int mult = 200;
	std::vector<my_pair> times(mult);
	for (int k = 0; k < count; k++)
	{
		for (int i = 1; i <= mult; i++)
		{
			auto pair = Cycle(out, i);
			times[i - 1].el_count = pair.el_count;
			times[i - 1].time += pair.time;
		}
	}

	for (int i = 0; i < mult; i++)
		times[i].time /= count;

	for (int i = 0; i < mult; i++)
		out << times[i].el_count << " " << times[i].time << std::endl;


	out.close();
	return 0;
}
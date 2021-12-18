#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "Element.h"
#include "Point.h"

using namespace std;

void Input(
	string pointsFile, vector<Point>& points,
	string weightsFile, vector<float>& q, vector<float>& p,
	string trianglesFile, vector<Element>& elements)
{
	// ¬вод вершин
	int nodeCount = 0;

	ifstream in(pointsFile);
	in >> nodeCount;
	points.resize(nodeCount);

	for (int i = 0; i < nodeCount; i++)
		in >> points[i].x >> points[i].y >> points[i].z;
	in.close();

	// ¬вод весов q и p
	in.open(weightsFile);

	int qCount = 0;
	in >> qCount;
	q.resize(qCount);

	int pCount = 0;
	in >> pCount;
	p.resize(pCount);

	for (int i = 0; i < qCount; i++)
		in >> q[i];

	for (int i = 0; i < pCount; i++)
		in >> p[i];

	in.close();

	// ¬вод конечного элемента (соответствующих номеров точек и весов q и p в массивах)
	in.open(trianglesFile);

	int edgeCount = 0;
	in >> edgeCount;

	for (int i = 0; i < edgeCount; i++)
	{
		int elementCount = 0;
		in >> elementCount;

		for (int j = 0; j < elementCount; j++)
		{
			Element el;

			in >> el.v1
				>> el.v2
				>> el.v3
				>> el.q1
				>> el.q2
				>> el.q3
				>> el.p;

			el.p -= qCount;

			elements.push_back(el);
		}
	}

	in.close();
}
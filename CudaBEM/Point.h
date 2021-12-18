#pragma once

#include "cuda_runtime.h"

struct Point
{
	float x, y, z;

	__host__ __device__  Point(float x, float y, float z) : x(x), y(y), z(z) {}
	__host__ __device__  Point() : x(0.0), y(0.0), z(0.0) {}

	__host__ __device__ Point operator+(const Point& a)
	{
		return Point(x + a.x, y + a.y, z + a.z);
	}

	__host__ __device__ Point operator-(const Point& a)
	{
		return Point(x - a.x, y - a.y, z - a.z);
	}

	__host__ __device__ float operator*(const Point& a)
	{
		return x * a.x + y * a.y + z * a.z;
	}

	__host__ __device__ Point operator*(float constant)
	{
		return Point(x * constant, y * constant, z * constant);
	}

	__host__ __device__ Point operator/(float constant)
	{
		return Point(x / constant, y / constant, z / constant);
	}

	__host__ __device__ float Norm()
	{
		return sqrt(Point(x, y, z) * Point(x, y, z));
	}

	__host__ __device__ Point Normalize()
	{
		Point v(x, y, z);
		return v / v.Norm();
	}

	__host__ __device__ Point Cross(const Point& v)
	{
		return
		{
			(y * v.z) - (z * v.y),
			(z * v.x) - (x * v.z),
			(x * v.y) - (y * v.x)
		};
	}
};

typedef Point Vector;

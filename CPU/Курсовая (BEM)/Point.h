#pragma once
struct Point
{
	Point(double x, double y, double z) : x(x), y(y), z(z) {}
	Point() : x(0.0), y(0.0), z(0.0) {}
	double x, y, z;

	Point operator+(Point a)
	{
		return Point(x + a.x, y + a.y, z + a.z);
	}

	Point operator-(Point a)
	{
		return Point(x - a.x, y - a.y, z - a.z);
	}

	double operator*(Point a)
	{
		return x * a.x + y * a.y + z * a.z;
	}

	Point operator*(double constant)
	{
		return Point(x * constant, y * constant, z * constant);
	}

	Point operator/(double constant)
	{
		return Point(x / constant, y / constant, z / constant);
	}

	double Norm()
	{
		return sqrt(Point(x, y, z) * Point(x, y, z));
	}

	Point Normalize()
	{
		Point v(x, y, z);
		return v / v.Norm();
	}

	Point Cross(Point v)
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

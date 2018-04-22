#ifndef ZAHNNN_MATH_H
#define ZAHNNN_MATH_H

namespace ZahnNN{

	template <typename T>
	inline T map(T x, T omin, T omax, T tmin, T tmax){
		return (x - omin) / (omax - omin) * (tmax - tmin) + tmin;
	}

	inline double to_n_decimals(double x, unsigned n){
		return int(x * pow(10, n)) / double(pow(10, n));
	}

}

#endif
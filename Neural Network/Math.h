#include <math.h>

namespace ZahnAI{

	template <typename T>
	inline T map(T x, T omin, T omax, T tmin, T tmax){
		return (x - omin) / (omax - omin) * (tmax - tmin) + tmin;
	}

	inline double to_n_decimals(double x, unsigned n){
		return int(x * pow(10, n)) / double(pow(10, n));
	}

}
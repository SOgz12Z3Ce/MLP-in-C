#include <math.h>

/**
 * @brief  sigmoid 函数
 * @param  x 自变量
 * @return sigmoid(x)
 */
static inline double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief  sigmoid 函数的导函数
 * @param  x 自变量
 * @return sigmoid'(x)
 */
static inline double d_sigmoid(double x)
{
	double sx = sigmoid(x);
	return sx * (1.0 - sx);
}

/**
 * @brief  恒等函数
 * @param  x 自变量
 * @return id(x)
 */
static inline double id(double x)
{
	return x;
}

/**
 * @brief  恒等函数的导函数
 * @param  x 自变量
 * @return id'(x)
 */
static inline double d_id(double x)
{
	return 1.0;
}

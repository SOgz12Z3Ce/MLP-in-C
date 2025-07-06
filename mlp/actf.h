#include <math.h>

/**
 * @brief  sigmoid 函数
 * @param  x 自变量
 * @return sigmoid(x)
 */
static inline float sigmoid(float x)
{
	return 1.0 / (1.0 + expf(-x));
}

/**
 * @brief  sigmoid 函数的导函数
 * @param  x 自变量
 * @return sigmoid'(x)
 */
static inline float d_sigmoid(float x)
{
	float sx = sigmoid(x);
	return sx * (1.0 - sx);
}

/**
 * @brief  恒等函数
 * @param  x 自变量
 * @return id(x)
 */
static inline float id(float x)
{
	return x;
}

/**
 * @brief  恒等函数的导函数
 * @param  x 自变量
 * @return id'(x)
 */
static inline float d_id(float x)
{
	return 1.0;
}

/**
 * @brief  ReLU 函数
 * @param  x 自变量
 * @return ReLU(x)
 */
static inline float relu(float x)
{
	return x < 0 ? 0 : x;
}

/**
 * @brief  ReLU 函数的导函数
 * @param  x 自变量
 * @return ReLU'(x)
 */
static inline float d_relu(float x)
{
	return x < 0 ? 0 : 1;
}

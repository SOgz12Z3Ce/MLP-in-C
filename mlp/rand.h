#include <stdlib.h>

/**
 * @brief  生成均一分布的随机数
 * @param  min 最小值
 * @param  max 最大值
 * @return `[min, max)`中的随机数
 */
static inline double rand_uniform(double min, double max)
{
	return min + (double)rand() / (RAND_MAX + 1.0) * (max - min);  /* Powered by ChatGPT */
}

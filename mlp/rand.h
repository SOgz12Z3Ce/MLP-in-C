#include <stdlib.h>

/**
 * @brief  生成均一分布的随机数
 * @param  min 最小值
 * @param  max 最大值
 * @return `[min, max)`中的随机数
 */
static inline float rand_uniform(float min, float max)
{
	return min + (float)rand() / (RAND_MAX + 1.0) * (max - min);  /* Powered by ChatGPT */
}

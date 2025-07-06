#ifndef VECTOR_H_
#define VECTOR_H_

#include <stddef.h>
#include <stdbool.h>

typedef struct Vector Vector;

/***** Vector *****/

struct Vector {
	size_t size;  /* 长度 */
	float *val;  /* 值 */

	/**
	 * @brief 销毁`Vector`
	 */
	void (*free)(Vector *this);

	/**
	 * @brief 设置值
	 * @param size 新的长度
	 * @param val  `[IN]`新的值
	 */
	void (*set)(Vector *this, size_t size, float *val);

	/**
	 * @brief 清空
	 */
	void (*clear)(Vector *this);

	/**
	 * @brief 均一随机填充值
	 * @param min 最小值
	 * @param max 最大值
	 */
	void (*rand_uniform)(Vector *this, float min, float max);

	/**
	 * @brief 相加
	 * @param target `[IN]`另一`Vector`
	 */
	void (*add)(Vector *this, Vector *target);

	/**
	 * @brief 相减
	 * @param target `[IN]`另一`Vector`
	 */
	void (*sub)(Vector *this, Vector *target);

	/**
	 * @brief 数乘
	 * @param scalar 倍率
	 */
	void (*scale)(Vector *this, float scalar);

	/**
	 * @brief 为每一个值作用函数
	 * @param func 函数
	 */
	void (*map)(Vector *this, float (*func)(float));

	/**
	 * @brief  拷贝自身
	 * @return `[OWN]`拷贝
	 */
	Vector *(*copy)(Vector *this);

	/**
	 * @brief  是否有负值
	 * @return 若有负值，返回`true`；否则，返回`false`
	 */
	bool (*has_negative)(Vector *this);

	/**
	 * @brief  获取数字长度
	 * @return `[OWN]`数字长度数组
	 */
	size_t *(*len)(Vector *this, size_t dp);

	/**
	 * @brief 打印
	 * @param dp 小数精度
	 */
	void (*print)(Vector *this, size_t dp);
};

/**
 * @brief  创建`Vector`
 * @param  size 维度
 * @param  val  `[IN]`值，传入`NULL`以令初始值为`0`
 * @return `[OWN]``Vector`指针
 */
Vector *new_vector(size_t size, float *val);

#endif  /* VECTOR_H_ */

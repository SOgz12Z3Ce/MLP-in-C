#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>
#include "vector.h"

typedef struct Matrix Matrix;

/***** Matrix *****/

struct Matrix {
	size_t row;    /* 行数 */
	size_t col;    /* 列数 */
	Vector **val;  /* 列向量 */

	/**
	 * @brief 销毁`Matrix`
	 */
	void (*free)(Matrix *this);

	/**
	 * @brief 清空
	 */
	void (*clear)(Matrix *this);

	/** 
	 * @brief 均一随机填充值
	 * @param min 最小值
	 * @param max 最大值
	 */
	void (*rand_uniform)(Matrix *this, double min, double max);

	/**
	 * @brief 转置矩阵
	 */
	void (*transpose)(Matrix *this);

	/**
	 * @brief  作用于`Vector`
	 * @param  target `[INOUT]`作用的`Vector`
	 */
	void (*act)(Matrix *this, Vector *target);

	/**
	 * @brief  相加
	 * @param  target `[IN]`另一`Matrix`
	 */
	void (*add)(Matrix *this, Matrix *target);

	/**
	 * @brief  相减
	 * @param  target `[IN]`另一`Matrix`
	 */
	void (*sub)(Matrix *this, Matrix *target);

	/**
	 * @brief 数乘
	 * @param scalar 倍率
	 */
	void (*scale)(Matrix *this, double scalar);

	/**
	 * @brief  拷贝自身
	 * @return `[OWN]`拷贝
	 */
	Matrix *(*copy)(Matrix *this);

	/**
	 * @brief 打印
	 * @param dp 小数精度
	 */
	void (*print)(Matrix *this, size_t dp);
};

/**
 * @brief  创建`Matrix`
 * @param  row  行数
 * @param  col  列数
 * @param  val `[IN]`列向量，传入`NULL`以令初始值为`0`
 * @return `[OWN]``Matrix`指针
 */
Matrix *new_matrix(size_t row, size_t col, Vector **val);

/***** 其他 *****/

/**
 * @brief  外积矩阵
 * @param  v1 列向量
 * @param  v2 行向量
 * @return `[OWN]`外积结果
 */
Matrix *outer(Vector *v1, Vector *v2);

#endif  /* MATRIX_H_ */

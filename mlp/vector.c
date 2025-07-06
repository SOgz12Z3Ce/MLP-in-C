#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vector.h"
#include "rand.h"

/***** 声明 *****/
/*** 外部 ***/

Vector *new_vector(size_t size, float *val);
static void vector_free(Vector *this);
static void vector_set(Vector *this, size_t size, float *val);
static void vector_clear(Vector *this);
static void vector_rand_uniform(Vector *this, float min, float max);
static void vector_add(Vector *this, Vector *target);
static void vector_sub(Vector *this, Vector *target);
static void vector_scale(Vector *this, float scalar);
static void vector_map(Vector *this, float (*func)(float));
static Vector *vector_copy(Vector *this);
static bool vector_has_negative(Vector *this);
static size_t *vector_len(Vector *this, size_t dp);
static void vector_print(Vector *this, size_t dp);

/*** 内部 ***/

/**
 * @brief  获取一个`float`变量的打印长度
 * @param  x  `float`变量
 * @param  dp 小数精度
 * @return 长度
 */
static size_t float_len(float x, size_t dp);

/***** 实现 *****/
/*** 外部 ***/

Vector *new_vector(size_t size, float *val)
{
	float *this_val = (float*)calloc(size, sizeof(float));
	if (!this_val)
		goto fail;
	if (val)
		memcpy(this_val, val, sizeof(float) * size);

	Vector *this = (Vector*)malloc(sizeof(Vector));
	if (!this)
		goto fail;
	*this = (Vector) {
		.size = size,
		.val = this_val,

		.free = vector_free,
		.set = vector_set,
		.clear = vector_clear,
		.rand_uniform = vector_rand_uniform,
		.add = vector_add,
		.sub = vector_sub,
		.scale = vector_scale,
		.map = vector_map,
		.copy = vector_copy,
		.has_negative = vector_has_negative,
		.len = vector_len,
		.print = vector_print,
	};
	return this;
fail:
	printf("Memory not enough!");
	exit(1);
}

static void vector_free(Vector *this)
{
	free(this->val);
	free(this);
}

static void vector_set(Vector *this, size_t size, float *val)
{
	if (this->size != size) {
		this->size = size;
		free(this->val);
		this->val = (float*)calloc(size, sizeof(float));
		if (!this->val)
			goto fail;
	}
	memcpy(this->val, val, sizeof(float) * size);
	return;
fail:
	printf("Memory not enough!");
	exit(1);
}

static void vector_clear(Vector *this)
{
	memset(this->val, 0, sizeof(float) * this->size);
}

static void vector_rand_uniform(Vector *this, float min, float max)
{
	for (size_t i = 0; i < this->size; i++)
		this->val[i] = rand_uniform(min, max);
}

static void vector_add(Vector *this, Vector *target)
{
	for (size_t i = 0; i < this->size; i++)
		this->val[i] += target->val[i];
}

static void vector_sub(Vector *this, Vector *target)
{
	for (size_t i = 0; i < this->size; i++)
		this->val[i] -= target->val[i];
}

static void vector_scale(Vector *this, float scalar)
{
	for (size_t i = 0; i < this->size; i++)
		this->val[i] *= scalar;
}

static void vector_map(Vector *this, float (*func)(float))
{
	for (size_t i = 0; i < this->size; i++)
		this->val[i] = func(this->val[i]);
}

static Vector *vector_copy(Vector *this)
{
	return new_vector(this->size, this->val);
}

static bool vector_has_negative(Vector *this)
{
	for (int i = 0; i != this->size; i++)
		if (signbit(this->val[i]))
			return true;
	return false;
}

static size_t *vector_len(Vector *this, size_t dp)
{
	size_t *ret = (size_t*)calloc(this->size, sizeof(size_t));
	if (!ret)
		goto fail;
	bool* space = (bool*)calloc(this->size, sizeof(bool));
	if (!space)
		goto fail;

	/* 前导空格 */
	if (this->has_negative(this))
		for (size_t i = 0; i < this->size; i++)
			space[i] = !(signbit(this->val[i]));
	
	/* 计算长度 */
	for (size_t i = 0; i < this->size; i++)
		ret[i] = space[i] + float_len(this->val[i], dp);

	free(space);
	return ret;
fail:
	printf("Memory not enough!");
	exit(1);
}

static void vector_print(Vector *this, size_t dp)
{
	size_t *len = this->len(this, dp);
	
	size_t max_len = 0;
	for (size_t i = 0; i < this->size; i++)
		if (max_len < len[i])
			max_len = len[i];
	bool has_negative = this->has_negative(this);

	printf("┌%*s┐\n", (int)max_len, "");
	for (size_t i = 0; i != this->size; i++) {
		printf("│");
		if (has_negative && !(signbit(this->val[i])))
			printf(" ");
		printf("%.*lf%*s│\n", (int)dp, this->val[i],
		       (int)(max_len - len[i]), "");
	}
	printf("└%*s┘\n", (int)max_len, "");

	free(len);
}

/*** 内部 ***/
static size_t float_len(float x, size_t dp)
{
	size_t len = 0;
	if (signbit(x)) {
		len += 1;
		x = -x;
	}
	if (x < 1.0)
		len += 1;
	else
		len += floor(log10f(x)) + 1;
	return len + 1 + dp;
}

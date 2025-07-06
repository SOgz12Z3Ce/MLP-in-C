#include <math.h>
#include "vector.h"

/***** 声明 *****/
/*** 外部 ***/

float mse_loss(Vector *out, Vector *label);
Vector *d_mse_loss(Vector *out, Vector *label);
float ce_loss(Vector *out, Vector *label);
Vector *d_ce_loss(Vector *out, Vector *label);
float softmax_ce_loss(Vector *out, Vector *label);
Vector *d_softmax_ce_loss(Vector *out, Vector *label);

/*** 内部 ***/

/**
 * @brief  归一化指数函数
 * @param  x `[IN]`输入
 * @return `[OWN]`softmax 输出
 */
static Vector *softmax(Vector *x);

/***** 实现 *****/
/*** 外部 ***/

float mse_loss(Vector *out, Vector *label)
{
	float ret = 0.0;
	for (size_t i = 0; i < out->size; i++)
		ret += powf(out->val[i] - label->val[i], 2);
	return ret;
}

Vector *d_mse_loss(Vector *out, Vector *label)
{
	Vector *ret = new_vector(out->size, NULL);
	for (size_t i = 0; i < out->size; i++)
		ret->val[i] = 2 * (out->val[i] - label->val[i]);
	return ret;
}

float ce_loss(Vector *out, Vector *label)
{
	float ret = 0.0;
	for (size_t i = 0; i < out->size; i++)
		ret -= label->val[i] * logf(out->val[i]);
	return ret;
}

Vector *d_ce_loss(Vector *out, Vector *label)
{
	Vector *ret = new_vector(out->size, NULL);
	for (size_t i = 0; i < out->size; i++)
		ret->val[i] = label->val[i] * -1.0 / out->val[i];
	return ret;
}

float softmax_ce_loss(Vector *out, Vector *label)
{
	Vector *sm_out = softmax(out);
	float ret = ce_loss(sm_out, label);
	sm_out->free(sm_out);
	return ret;
}

Vector *d_softmax_ce_loss(Vector *out, Vector *label)
{
	Vector *sm_out = softmax(out);
	Vector* ret = new_vector(out->size, NULL);
	for (size_t i = 0; i < ret->size; i++)
		ret->val[i] = sm_out->val[i] - label->val[i];
	sm_out->free(sm_out);
	return ret;
}

/*** 内部 ***/

static Vector *softmax(Vector *x)
{
	Vector *ret = x->copy(x);
	ret->map(ret, expf);
	float base = 0.0;
	for (size_t i = 0; i < x->size; i++)
		base += ret->val[i];
	ret->scale(ret, 1.0 / base);
	return ret;
}

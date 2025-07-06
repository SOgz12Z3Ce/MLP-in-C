#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "matrix.h"
#include "mlp.h"
#include "rand.h"

/***** 声明 *****/
/*** 外部 ***/

FCLayer *new_fc_layer(size_t size, size_t next_size, Matrix *weight,
                      Vector *bias, float (*actf)(float),
                      float (*dactf)(float));
static void fc_layer_free(FCLayer *this);
static void fc_layer_clear(FCLayer *this);
static void fc_layer_forward(FCLayer *this, Vector *input);
static void fc_layer_add(FCLayer *this, FCLayer *target);
static void fc_layer_sub(FCLayer *this, FCLayer *target);
static void fc_layer_scale(FCLayer *this, float scalar);
static FCLayer *fc_layer_copy(FCLayer *this);

MLPNet *new_mlp_net(size_t size, FCLayer **layer,
                    float (*lossf)(Vector*, Vector*),
                    Vector *(*dlossf)(Vector*, Vector*));
static void mlp_net_free(MLPNet *this);
static void mlp_net_init_xavier(MLPNet *this);
static void mlp_net_forward(MLPNet *this, Vector *input);
static void mlp_net_grad(MLPNet *this, Vector *label, MLPGrad *grad);
static void mlp_net_update(MLPNet *this, MLPGrad *grad);

MLPGrad *new_mlp_grad(MLPNet *net);
static void mlp_grad_free(MLPGrad *this);
static void mlp_grad_clear(MLPGrad *this);
static void mlp_grad_add(MLPGrad *this, MLPGrad *target);
static void mlp_grad_scale(MLPGrad *this, float scalar);

/*** 内部 ***/

static void backward(FCLayer *net, FCLayer *grad, Vector *out_grad);

/***** 实现 *****/
/*** 外部 ***/

FCLayer *new_fc_layer(size_t size, size_t next_size, Matrix *weight,
                      Vector *bias, float (*actf)(float),
                      float (*dactf)(float))
{
	Vector *this_node = new_vector(size, NULL);
	Matrix *this_weight;
	if (weight)
		this_weight = weight->copy(weight);
	else
		this_weight = new_matrix(next_size, size, NULL);
	Vector *this_bias;
	if (bias)
		this_bias = bias->copy(bias);
	else
		this_bias = new_vector(next_size, NULL);
	Vector *this_pre = new_vector(next_size, NULL);
	Vector *this_out = new_vector(next_size, NULL);

	FCLayer *this = (FCLayer*)malloc(sizeof(FCLayer));
	if (!this)
		goto fail;
	*this = (FCLayer) {
		.size = size,
		.next_size = next_size,
		.node = this_node,
		.weight = this_weight,
		.bias = this_bias,
		.pre = this_pre,
		.out = this_out,
		.actf = actf,
		.dactf = dactf,

		.free = fc_layer_free,
		.clear = fc_layer_clear,
		.forward = fc_layer_forward,
		.add = fc_layer_add,
		.sub = fc_layer_sub,
		.scale = fc_layer_scale,
		.copy = fc_layer_copy,
	};
	return this;
fail:
	printf("Memory not enough!");
	exit(1);
}

static void fc_layer_free(FCLayer *this)
{
	this->node->free(this->node);
	this->weight->free(this->weight);
	this->bias->free(this->bias);
	this->pre->free(this->pre);
	this->out->free(this->out);
	free(this);
}

static void fc_layer_clear(FCLayer *this)
{
	this->node->clear(this->node);
	this->weight->clear(this->weight);
	this->bias->clear(this->bias);
	this->pre->clear(this->pre);
	this->out->clear(this->out);
}

static void fc_layer_forward(FCLayer *this, Vector *input)
{
	Vector *tmp = input->copy(input);
	this->node->set(this->node, this->size, tmp->val);
	this->weight->act(this->weight, tmp);
	tmp->add(tmp, this->bias);
	this->pre->set(this->pre, this->next_size, tmp->val);
	tmp->map(tmp, this->actf);
	this->out->set(this->out, this->next_size, tmp->val);
	tmp->free(tmp);
}

static void fc_layer_add(FCLayer *this, FCLayer *target)
{
	this->weight->add(this->weight, target->weight);
	this->bias->add(this->bias, target->bias);
}

static void fc_layer_sub(FCLayer *this, FCLayer *target)
{
	this->weight->sub(this->weight, target->weight);
	this->bias->sub(this->bias, target->bias);
}

static void fc_layer_scale(FCLayer *this, float scalar)
{
	this->weight->scale(this->weight, scalar);
	this->bias->scale(this->bias, scalar);
}

static FCLayer *fc_layer_copy(FCLayer *this)
{
	new_fc_layer(this->size, this->next_size, this->weight, this->bias,
	             this->actf, this->dactf);
}

MLPNet *new_mlp_net(size_t size, FCLayer **layer,
                    float (*lossf)(Vector*, Vector*),
                    Vector *(*dlossf)(Vector*, Vector*))
{
	FCLayer **this_layer = (FCLayer**)calloc(size, sizeof(FCLayer*));
	if (!this_layer)
		goto fail;
	for (size_t i = 0; i < size; i++)
		this_layer[i] = layer[i]->copy(layer[i]);
	
	MLPNet *this = (MLPNet*)malloc(sizeof(MLPNet));
	if (!this)
		goto fail;
	*this = (MLPNet) {
		.size = size,
		.layer = this_layer,
		.lossf = lossf,
		.dlossf = dlossf,

		.free = mlp_net_free,
		.init_xavier = mlp_net_init_xavier,
		.forward = mlp_net_forward,
		.grad = mlp_net_grad,
		.update = mlp_net_update,
	};
	return this;
fail:
	printf("Memory not enough!");
	exit(1);
}

static void mlp_net_free(MLPNet *this)
{
	for (size_t i = 0; i < this->size; i++)
		this->layer[i]->free(this->layer[i]);
	free(this);
}

static void mlp_net_init_xavier(MLPNet *this)
{
	for (size_t i = 0; i < this->size; i++) {
		FCLayer *layer = this->layer[i];
		float bound = sqrt(6.0 / (layer->size + layer->next_size));
		layer->weight->rand_uniform(layer->weight, -bound , bound);
		layer->bias->clear(layer->bias);
	}
}

static void mlp_net_forward(MLPNet *this, Vector *input)
{
	for (size_t i = 0; i < this->size; i++) {
		FCLayer *layer = this->layer[i];
		layer->forward(layer, input);
		input = layer->out;	
	}
}

static void mlp_net_grad(MLPNet *this, Vector *label, MLPGrad *grad)
{
	Vector *out = this->layer[this->size - 1]->out;
	Vector *out_grad = this->dlossf(out, label);
	Vector *tmp = out_grad;
	for (size_t i = this->size; i-- > 0; ) {
		backward(this->layer[i], grad->layer[i], out_grad);
		out_grad = grad->layer[i]->node;
	}
	tmp->free(tmp);
}

static void mlp_net_update(MLPNet *this, MLPGrad *grad)
{
	for (size_t i = 0; i < this->size; i++)
		this->layer[i]->sub(this->layer[i], grad->layer[i]);
}

MLPGrad *new_mlp_grad(MLPNet *net)
{
	size_t this_size = net->size;
	FCLayer **this_layer = (FCLayer**)calloc(net->size, sizeof(FCLayer*));
	if (!this_layer)
		goto fail;
	for (size_t i = 0; i < net->size; i++)
		this_layer[i] = net->layer[i]->copy(net->layer[i]);
	
	MLPGrad *this = (MLPGrad*)malloc(sizeof(MLPGrad));
	if (!this)
		goto fail;
	*this = (MLPGrad) {
		.size = this_size,
		.layer = this_layer,

		.free = mlp_grad_free,
		.clear = mlp_grad_clear,
		.add = mlp_grad_add,
		.scale = mlp_grad_scale,
	};
	return this;
fail:
	printf("Memory not enough!");
	exit(1);
}

static void mlp_grad_free(MLPGrad *this)
{
	for (size_t i = 0; i < this->size; i++)
		this->layer[i]->free(this->layer[i]);
	free(this);
}

static void mlp_grad_clear(MLPGrad *this)
{
	for (size_t i = 0; i < this->size; i++)
		this->layer[i]->clear(this->layer[i]);
}

static void mlp_grad_add(MLPGrad *this, MLPGrad *target)
{
	for (size_t i = 0; i < this->size; i++)
		this->layer[i]->add(this->layer[i], target->layer[i]);
}

static void mlp_grad_scale(MLPGrad *this, float scalar)
{
	for (size_t i = 0; i < this->size; i++)
		this->layer[i]->scale(this->layer[i], scalar);
}

/*** 内部 ***/

/**
 * @brief 反向传播
 * @param net      `[IN]`网络层
 * @param grad     `[INOUT]`梯度层
 * @param out_grad `[IN]`输出层梯度
 */
static void backward(FCLayer *net, FCLayer *grad, Vector *out_grad)
{
	Vector *tmp_v;
	Matrix *tmp_m;
	grad->out->set(grad->out, out_grad->size, out_grad->val);

	/***** pre *****/
	tmp_v = net->pre->copy(net->pre);
	tmp_v->map(tmp_v, net->dactf);
	for (size_t i = 0; i < net->next_size; i++)
		grad->pre->val[i] = tmp_v->val[i] * (grad->out->val[i]);
	tmp_v->free(tmp_v);

	/***** bias *****/
	grad->bias->free(grad->bias);
	grad->bias = grad->pre->copy(grad->pre);

	/***** weight *****/
	grad->weight->free(grad->weight);
	grad->weight = outer(grad->pre, net->node);

	/***** node *****/
	tmp_m = net->weight->copy(net->weight);
	tmp_v = grad->pre->copy(grad->pre);
	tmp_m->transpose(tmp_m);
	tmp_m->act(tmp_m, tmp_v);
	tmp_m->free(tmp_m);
	grad->node->free(grad->node);
	grad->node = tmp_v;
}

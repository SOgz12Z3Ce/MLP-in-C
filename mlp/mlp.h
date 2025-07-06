#ifndef MLP_H_
#define MLP_H_

#include <stddef.h>
#include <math.h>
#include "vector.h"
#include "matrix.h"

typedef struct FCLayer FCLayer;
typedef struct MLPNet MLPNet;
typedef struct MLPGrad MLPGrad;

/***** FCLayer *****/

struct FCLayer {
	size_t size;       /* 大小 */
	size_t next_size;  /* 下层大小 */
	Vector *node;      /* 节点 */
	Matrix *weight;    /* 权重 */
	Vector *bias;      /* 偏置 */
	Vector *pre;       /* 线性变换结果 */
	Vector *out;       /* 输出 */
	float (*actf)(float x);   /* 激活函数 */
	float (*dactf)(float x);  /* 激活函数的导函数 */

	/**
	 * @brief 销毁`FCLayer`
	 */
	void (*free)(FCLayer *this);

	/**
	 * @brief 清空
	 */
	void (*clear)(FCLayer *this);

	/**
	 * @brief  前向传播
	 * @param  input `[IN]`输入
	 */
	void (*forward)(FCLayer *this, Vector *input);

	/**
	 * @brief  相加
	 * @param  target `[IN]`另一`FCLayer`
	 */
	void (*add)(FCLayer *this, FCLayer *target);

	/**
	 * @brief  相减
	 * @param  target `[IN]`另一`FCLayer`
	 */
	void (*sub)(FCLayer *this, FCLayer *target);

	/**
	 * @brief 数乘
	 * @param scalar 倍率
	 */
	void (*scale)(FCLayer *this, float scalar);

	/**
	 * @brief  拷贝自身
	 * @return `[OWN]`拷贝
	 */
	FCLayer *(*copy)(FCLayer *this);
};

/**
 * @brief  创建`FCLayer`
 * @param  size      大小
 * @param  next_size 下层大小
 * @param  weight    `[IN]`权重，传入`NULL`以令初始值为`0`
 * @param  bias      `[IN]`偏置，传入`NULL`以令初始值为`0`
 * @param  actf      激活函数
 * @param  dactf     激活函数的导函数
 * @return `FCLayer`指针
 */
FCLayer *new_fc_layer(size_t size, size_t next_size, Matrix *weight,
                      Vector *bias, float (*actf)(float),
                      float (*dactf)(float));

/***** MLPNet *****/

struct MLPNet {
	size_t size;      /* 不含输出层的层数 */
	FCLayer **layer;  /* 层 */
	float (*lossf)(Vector*, Vector*);    /* 损失函数 */
	Vector *(*dlossf)(Vector*, Vector*);  /* 损失函数的梯度函数 */

	/**
	 * @brief 销毁 MLPNet
	 */
	void (*free)(MLPNet *this);

	/**
	 * @brief Xavier 初始化网络
	 */
	void (*init_xavier)(MLPNet *this);

	/** 
	 * @brief 前向传播
	 * @param input `[IN]`输入
	 */
	void (*forward)(MLPNet *this, Vector *input);

	/**
	 * @brief 计算梯度
	 * @param label `[IN]`标签
	 * @param grad  `[OUT]`梯度容器
	 */
	void (*grad)(MLPNet *this, Vector *label, MLPGrad *grad);

	/**
	 * @brief 更新参数
	 * @param grad 梯度
	 */
	void (*update)(MLPNet *this, MLPGrad *grad);
};

/**
 * @brief 创建`MLPNet`
 * @param size  含输出层的层数
 * @param layer `[IN]`层
 * @param loss  损失函数
 * @param dloss 损失函数的导函数
 */
MLPNet *new_mlp_net(size_t size, FCLayer **layer,
                    float (*lossf)(Vector*, Vector*),
                    Vector *(*dlossf)(Vector*, Vector*));

/***** MLPGrad *****/

struct MLPGrad
{
	size_t size;      /* 不含输出层的层数 */
	FCLayer **layer;  /* 层 */

	/**
	 * @brief 销毁`MLPGrad`
	 */
	void (*free)(MLPGrad *this);

	/**
	 * @brief 清空
	 */
	void (*clear)(MLPGrad *this);

	/**
	 * @brief 相加
	 * @param target `[IN]`一`MLPGrad`
	 */
	void (*add)(MLPGrad *this, MLPGrad *target);

	/**
	 * @brief 数乘
	 * @param scalar 倍率
	 */
	void (*scale)(MLPGrad *this, float scalar);
};

/**
 * @brief  创建`MLPGrad`
 * @param  net `[IN]`对应的`MLPNet`
 * @return `[OWN]``MLPGrad`指针
 */
MLPGrad *new_mlp_grad(MLPNet *net);

#endif  /* MLP_H_ */

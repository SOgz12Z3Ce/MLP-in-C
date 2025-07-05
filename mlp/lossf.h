#include "vector.h"

/**
 * @brief 平方差损失函数
 * @param  out   `[IN]`网络输出层
 * @param  label `[IN]`输出层标签
 * @return 损失值
 */
double mse_loss(Vector *out, Vector *label);

/**
 * @brief  计算输出层梯度（平方差损失函数）
 * @param  out   `[IN]`网络输出层
 * @param  label `[IN]`输出层标签
 * @return `[OWN]`输出层梯度
 */
Vector *d_mse_loss(Vector *out, Vector *label);

/**
 * @brief  交叉熵损失函数
 * @param  out   `[IN]`网络输出层
 * @param  label `[IN]`输出层标签
 * @return 损失值
 */
double ce_loss(Vector *out, Vector *label);

/**
 * @brief  计算输出层梯度（交叉熵损失函数）
 * @param  out   `[IN]`网络输出层
 * @param  label `[IN]`输出层标签
 * @return `[OWN]`输出层梯度
 */
Vector *d_ce_loss(Vector *out, Vector *label);

/**
 * @brief  归一化指数函数 + 交叉熵损失函数
 * @param  out   `[IN]`网络输出层
 * @param  label `[IN]`输出层标签
 * @return 损失值
 */
double softmax_ce_loss(Vector *out, Vector *label);

/**
 * @brief  计算输出层梯度（归一化指数函数 + 交叉熵损失函数）
 * @param  out   `[IN]`网络输出层
 * @param  label `[IN]`输出层标签
 * @return `[OWN]`输出层梯度
 */
Vector *d_softmax_ce_loss(Vector *out, Vector *label);

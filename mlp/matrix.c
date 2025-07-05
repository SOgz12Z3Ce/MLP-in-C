#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "matrix.h"

/***** 声明 *****/

Matrix *new_matrix(size_t row, size_t col, Vector **val);
static void matrix_free(Matrix *this);
static void matrix_clear(Matrix *this);
static void matrix_rand_uniform(Matrix *this, double min, double max);
static void matrix_transpose(Matrix *this);
static void matrix_act(Matrix *this, Vector *target);
static void matrix_add(Matrix *this, Matrix *target);
static void matrix_sub(Matrix *this, Matrix *target);
static void matrix_scale(Matrix *this, double scalar);
static Matrix *matrix_copy(Matrix *this);
static void matrix_print(Matrix *this, size_t dp);

Matrix *outer(Vector *v1, Vector *v2);

/***** 实现 *****/

Matrix *new_matrix(size_t row, size_t col, Vector **val)
{
	Vector **this_val = (Vector**)calloc(col, sizeof(Vector*));
	if (!this_val)
		goto fail;
	if (val)
		for (size_t i = 0; i < col; i++)
			this_val[i] = val[i]->copy(val[i]);
	else
		for (size_t i = 0; i < col; i++)
			this_val[i] = new_vector(row, NULL);
	
	Matrix *this = (Matrix*)malloc(sizeof(Matrix));
	if (!this)
		goto fail;
	*this = (Matrix) {
		.row = row,
		.col = col,
		.val = this_val,

		.free = matrix_free,
		.clear = matrix_clear,
		.rand_uniform = matrix_rand_uniform,
		.transpose = matrix_transpose,
		.act = matrix_act,
		.add = matrix_add,
		.sub = matrix_sub,
		.scale = matrix_scale,
		.copy = matrix_copy,
		.print = matrix_print,
	};
	return this;
fail:
	printf("Memory not enough!");
	exit(1);
}

void matrix_free(Matrix *this)
{
	for (size_t i = 0; i < this->col; i++)
		this->val[i]->free(this->val[i]);
	free(this->val);
	free(this);
}

void matrix_clear(Matrix *this)
{
	for (size_t i = 0; i < this->col; i++)
		this->val[i]->clear(this->val[i]);
}

void matrix_rand_uniform(Matrix *this, double min, double max)
{
	for (size_t i = 0; i < this->col; i++)
		this->val[i]->rand_uniform(this->val[i], min, max);
}

void matrix_transpose(Matrix *this)
{
	size_t row = this->row;
	size_t col = this->col;

	Vector **new_val = (Vector**)calloc(row, sizeof(Vector*));
	if (!new_val)
		goto fail;
	for (size_t i = 0; i < row; i++) {
		new_val[i] = new_vector(col, NULL);
		for (size_t j = 0; j < col; j++)
			new_val[i]->val[j] = this->val[j]->val[i];
	}
	for (size_t i = 0; i < col; i++)
		this->val[i]->free(this->val[i]);
	free(this->val);
	
	this->row = col;
	this->col = row;
	this->val = new_val;
	return;
fail:
	printf("Memory not enough!");
	exit(1);
}

static void matrix_act(Matrix *this, Vector *target)
{
	Vector *res = new_vector(this->row, NULL);
	Vector *tmp;
	for (size_t i = 0; i < this->col; i++) {
		tmp = this->val[i]->copy(this->val[i]);
		tmp->scale(tmp, target->val[i]);
		res->add(res, tmp);
		tmp->free(tmp);
	}
	target->set(target, this->row, res->val);
	res->free(res);
}

static void matrix_add(Matrix *this, Matrix *target)
{
	for (size_t i = 0; i < this->col; i++)
		this->val[i]->add(this->val[i], target->val[i]);
}

static void matrix_sub(Matrix *this, Matrix *target)
{
	for (size_t i = 0; i < this->col; i++)
		this->val[i]->sub(this->val[i], target->val[i]);
}

static void matrix_scale(Matrix *this, double scalar)
{
	for (size_t i = 0; i < this->col; i++)
		this->val[i]->scale(this->val[i], scalar);
}

static Matrix *matrix_copy(Matrix *this)
{
	return new_matrix(this->row, this->col, this->val);
}

static void matrix_print(Matrix *this, size_t dp)
{
	size_t **len = (size_t**)calloc(this->col, sizeof(size_t*));
	if (!len)
		goto fail;
	for (size_t i = 0; i < this->col; i++)
		len[i] = this->val[i]->len(this->val[i], dp);
	
	size_t *max_len = (size_t*)calloc(this->col, sizeof(size_t));
	if (!max_len)
		goto fail;
	for (size_t i = 0; i < this->col; i++) {
		size_t tmp = 0;
		for (size_t j = 0; j < this->row; j++)
			if (len[i][j] > tmp)
				tmp = len[i][j];
		max_len[i] = tmp;
	}

	bool *has_negative = (bool*)calloc(this->col, sizeof(bool));
	if (!has_negative)
		goto fail;
	for (size_t i = 0; i < this->col; i++)
		has_negative[i] = this->val[i]->has_negative(this->val[i]);
	
	size_t print_len = this->col - 1;
	for (size_t i = 0; i < this->col; i++)
		print_len += max_len[i];
	printf("┌%*s┐\n", (int)print_len, "");
	for (size_t i = 0; i < this->row; i++) {
		printf("│");
		for (size_t j = 0; j < this->col; j++) {
			if (has_negative && !(signbit(this->val[j]->val[i])))
				printf(" ");
			printf("%.*lf%*s", (int)dp, this->val[j]->val[i],
			       (int)max_len[i], "");
			if (j + 1 != this->col)
				printf(" ");
		}
		printf("│\n");
	}
	printf("└%*s┘\n", (int)print_len, "");

	for (size_t i = 0; i < this->col; i++)
		free(len[i]);
	free(len);
	free(max_len);
	free(has_negative);
	return;
fail:
	printf("Memory not enough!");
	exit(1);
}

Matrix *outer(Vector *v1, Vector *v2)
{
	size_t row = v1->size;
	size_t col = v2->size;
	Vector **ret_val = (Vector**)calloc(col, sizeof(Vector*));
	for (size_t i = 0; i < col; i++) {
		Vector *tmp;
		tmp = v1->copy(v1);
		tmp->scale(tmp, v2->val[i]);
		ret_val[i] = tmp;
	}

	Matrix *ret = new_matrix(row, col, ret_val);
	return ret;
}

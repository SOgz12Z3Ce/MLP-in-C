#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "vector.h"
#include "mlp.h"
#include "actf.h"
#include "lossf.h"

Vector **read_image_file(char *path);
Vector **read_label_file(char *path);
uint32_t read_uint32(FILE *file);
uint8_t read_uint8(FILE *file);
Vector *read_image(FILE *file, size_t size);
Vector *read_label(FILE *file);
size_t *rand_queue();
size_t res(Vector *out);

#define TRAIN_SIZE 60000
#define BATCH_SIZE 100
#define BATCH_NUM TRAIN_SIZE / BATCH_SIZE - 1
#define LEARNING_RATE 5
#define TEST_SIZE 10000

#define NET_SIZE 4
size_t layer_size[NET_SIZE] = {784, 16, 16, 10};

int main()
{
	srand(time(NULL));
	Vector **train_image = read_image_file("../mnist/train-images.idx3-ubyte");
	Vector **train_label = read_label_file("../mnist/train-labels.idx1-ubyte");
	size_t *queue = rand_queue();

	printf("Training start.\n");
	printf("Batch size: %d\n", BATCH_SIZE);
	printf("number of batch(es): %d(drop last)\n\n", BATCH_NUM);
	
	FCLayer *hidden_layer_1 = new_fc_layer(layer_size[0], layer_size[1],
	                                       NULL, NULL, sigmoid, d_sigmoid);
	FCLayer *hidden_layer_2 = new_fc_layer(layer_size[1], layer_size[2],
	                                       NULL, NULL, sigmoid, d_sigmoid);
	FCLayer *output_layer = new_fc_layer(layer_size[2], layer_size[3],
	                                     NULL, NULL, sigmoid, d_sigmoid);
	FCLayer *layer[3] = {hidden_layer_1, hidden_layer_2, output_layer};
	MLPNet *net = new_mlp_net(NET_SIZE - 1, layer,
	                          mse_loss, d_mse_loss);
	MLPGrad *grad_tmp = new_mlp_grad(net);
	MLPGrad *grad = new_mlp_grad(net);
	net->init_xavier(net);

	for (size_t i = 0; i < BATCH_NUM; i++) {
		for (size_t j = 0; j < BATCH_SIZE; j++) {
			size_t index = queue[i * BATCH_SIZE + j];
			Vector *input = train_image[index];
			Vector *label = train_label[index];
			net->forward(net, input);
			net->grad(net, label, grad_tmp);
			grad->add(grad, grad_tmp);
		}
		grad->scale(grad, 1.0 / BATCH_SIZE * LEARNING_RATE);
		net->update(net, grad);
		grad->clear(grad);

		net->forward(net, train_image[0]);
		Vector *out = net->layer[net->size - 1]->out;
		double loss = net->lossf(out, train_label[0]);
		printf("[%d / %d] loss: %lf\n", (int)i + 1, BATCH_NUM, loss);
	}
	printf("Done.\n\n");

	Vector **test_image = read_image_file("../mnist/t10k-images.idx3-ubyte");
	Vector **test_label = read_label_file("../mnist/t10k-labels.idx1-ubyte");
	int correct = 0;
	printf("Testing start.\n");
	for (size_t i = 0; i < TEST_SIZE; i++) {
		Vector *input = test_image[i];
		Vector *label = test_label[i];
		net->forward(net, input);
		if (label->val[res(net->layer[net->size - 1]->out)] == 1.0)
			correct += 1;
	}
	printf("Done.\n");
	printf("Accuracy: %%%.2lf (%d / %d)\n", (double)correct / TEST_SIZE * 100, correct, TEST_SIZE);

	printf("\n----- end of program -----\n");
	return 0;
}

Vector **read_image_file(char *path)
{
	printf("Reading: %s\n", path);
	FILE *file = fopen(path, "rb");
	fseek(file, 0, SEEK_SET);

	uint32_t magic_num = read_uint32(file);
	printf("Magic number: 0x%08X\n", magic_num);

	uint32_t image_num = read_uint32(file);
	printf("Number of image(s): %u\n", image_num);
	
	uint32_t image_row = read_uint32(file);
	printf("Rows of image(s): %u\n", image_row);

	uint32_t image_col = read_uint32(file);
	printf("Cols of image(s): %u\n", image_col);

	printf("Start to read image(s)...\n");
	Vector **ret = (Vector**)calloc(image_num, sizeof(Vector*));
	for (size_t i = 0; i < image_num; i++)
		ret[i] = read_image(file, image_row * image_col);
	printf("Done.\n\n");

	fclose(file);
	return ret;
}

Vector **read_label_file(char *path)
{
	printf("Reading: %s\n", path);
	FILE *file = fopen(path, "rb");
	fseek(file, 0, SEEK_SET);

	uint32_t magic_num = read_uint32(file);
	printf("Magic number: 0x%08X\n", magic_num);
	
	uint32_t label_num = read_uint32(file);
	printf("Number of label(s): %u\n", label_num);

	printf("Start to read label(s)...\n");
	Vector **ret = (Vector**)calloc(label_num, sizeof(Vector*));
	for (size_t i = 0; i < label_num; i++)
		ret[i] = read_label(file);
	printf("Done.\n\n");

	fclose(file);
	return ret;
}

uint32_t read_uint32(FILE *file)
{
	uint32_t ret = 0;
	uint8_t tmp;
	for (size_t i = 0; i < 4; i++) {
		fread(&tmp, sizeof(uint8_t), 1, file);
		ret = ret << 8;
		ret += tmp;
	}
	return ret;
}

uint8_t read_uint8(FILE *file)
{
	uint8_t ret;
	fread(&ret, sizeof(uint8_t), 1, file);
	return ret;
}

Vector *read_image(FILE *file, size_t size)
{
	Vector *ret = new_vector(size, NULL);
	for (size_t i = 0; i < size; i++)
		ret->val[i] = (double)read_uint8(file) / 255;
	return ret;
}

Vector *read_label(FILE *file)
{
	uint8_t label = read_uint8(file);
	Vector *ret = new_vector(10, NULL);
	ret->val[label] = 1.0;
	return ret;
}

size_t *rand_queue() {
	size_t *ret = (size_t*)calloc(TRAIN_SIZE, sizeof(size_t));
	for (size_t i = 0; i != TRAIN_SIZE; i++)
		ret[i] = i;
	for (size_t i = TRAIN_SIZE; i-- > 0; ) {
		size_t j = rand() % (i + 1);
		size_t tmp = ret[i];
		ret[i] = ret[j];
		ret[j] = tmp;
	}
	return ret;
}

size_t res(Vector *out)
{
	double max = -100.0;
	size_t max_index = 0;
	for (size_t i = 0; i < out->size; i++) {
		if (out->val[i] > max) {
			max = out->val[i];
			max_index = i;
		}
	}
	return max_index;
}

# MLP in C

仅使用标准库制作的多层感知器。

## 使用

- `vector.h` `Matrix.h`提供了基本的数学对象。
- `mlp.h`提供了网络对象。
- `actf.h` `lossf.h` `rand.h`提供了一些数学方法。

具体用法见文件内注释。

要在项目中使用，需指定`mlp`为 include 路径并添加到源文件列表，例如：
```cmake
include_directories(path/to/mlp)
aux_source_directory(path/to/mlp SRC_LIST)
```

## 示例

`example/`下提供了一个识别`mnist`数据集的示例，使用以下方法构建：
```bash
mkdir build
cd build
cmake ..
make
```
运行`example/bin/demo`可训练网络并测试效果。

## 语法风格

整体遵循 kernel 风格。

使用了面向对象风格的写法，即`struct`模拟对象，函数指针模拟方法，手动传递`this`指针。
例如：
```c
/* vector.h */
typedef struct Vector Vector

struct Vector {
	size_t size;  /* 长度 */
	float *val;  /* 值 */

	/**
	 * @brief 销毁`Vector`
	 */
	void (*free)(Vector *this);
	
	/**
	 * @brief 相加
	 * @param target `[IN]`另一`Vector`
	 */
	void (*add)(Vector *this, Vector *target);

	/**
	 * @brief 打印
	 */
	void (*print)(Vector *this, size_t dp);
}

/**
 * @brief  创建`Vector`
 * @param  size 维度
 * @param  val  `[IN]`值，传入`NULL`以令初始值为`0`
 * @return `[OWN]``Vector`指针
 */
Vector *new_vector(size_t size, float *val);
```
```c
/* example.c */
#include "vector.h"
int main()
{
	float val1[2] = {-1.0, 2.0}
	Vector *v1 = new_vector(2, val1);


	float val2[2] = {-2.0, 0.0}
	Vector *v2 = new_vector(2, val2);

	v1->add(v1, v2);
	v1->print(v1, 1);  /* [-3.0, 2.0] */

	return 0;
}

```

# 鸣谢
感谢[3Blue1Brown 的深度学习系列](https://space.bilibili.com/88461692/lists/1528929?type=series)。

感谢某两个人，虽然没帮我写代码但对我很好。（他们看到了自会知道说的是自己）

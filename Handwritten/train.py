import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 定义模型保存路径
model_path = "saved_model/mnist_model.h5"

# 加载和预处理数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 检查模型文件是否存在
if os.path.exists(model_path):
    # 直接加载已保存的模型
    model = tf.keras.models.load_model(model_path)
    print("模型已加载。")
else:
    # 构建模型
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    # 定义 EarlyStopping 回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # 定义 ReduceLROnPlateau 回调   
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # 在训练时同时使用这两个回调
    history = model.fit(x_train, y_train, 
                        epochs=50, 
                        validation_split=0.1, 
                        callbacks=[early_stopping, reduce_lr])

    # 保存模型
    model.save(model_path)
    print(f"模型已保存至 {model_path}")

# 评估模型性能
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n测试集上的准确率: {test_acc:.2f}')

# 进行预测
sample_image = x_test[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample_image)
print(f"预测结果: {prediction.argmax()}")
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"预测: {prediction.argmax()}")
plt.show()

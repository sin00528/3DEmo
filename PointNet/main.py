import keras
from keras.callbacks import ModelCheckpoint
from model import build_model
import numpy as np
import matplotlib.pyplot as plt

IN_DIR = "data/"

RND_SEED = 0
BATCH_SIZE = 64
EPOCHS = 20

# 데이터 로드
train_X = np.load(open(IN_DIR + 'train_X.npy', 'rb'))
val_X = np.load(open(IN_DIR + 'val_X.npy', 'rb'))
test_X = np.load(open(IN_DIR + 'test_X.npy', 'rb'))

train_y = np.load(open(IN_DIR + 'train_y.npy', 'rb'))
val_y = np.load(open(IN_DIR + 'val_y.npy', 'rb'))
test_y = np.load(open(IN_DIR + 'test_y.npy', 'rb'))

# 모델 로드
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 콜백 함수 등록
check_point = ModelCheckpoint(filepath='./logs/weights.h5', monitor='val_loss', verbose=1, save_best_only=True)

# 모델 훈련
history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(val_X, val_y))
                    #validation_data=(val_X, val_y)), callbacks=[check_point])

# 정확도 출력
results = model.evaluate(test_X, test_y)
print('Test accuracy: ', results[1])

# 그래프 출력
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('{} epochs, Accuracy : {:.6f}'.format(EPOCHS, results[1]))
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.savefig('acc.png')
plt.show()

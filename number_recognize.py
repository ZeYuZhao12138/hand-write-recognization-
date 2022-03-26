# 2022/03/20 23:22
# have a good day!
import sys
import pygame
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.layers import AveragePooling2D
from keras.layers import Flatten
import numpy as np


def pencil_point(center, size):
    point1 = (center[0] - size[0] / 2, center[1] - size[1] / 2)
    point2 = (center[0] - size[0] / 2, center[1] + size[1] / 2)
    point3 = (center[0] + size[0] / 2, center[1] - size[1] / 2)
    point4 = (center[0] + size[0] / 2, center[1] + size[1] / 2)
    return point1, point2, point3, point4


def possible_bar(possibility):
    length = []
    for n in range(10):
        length.append(possibility[0][n] * 140)
    return length


def load_data():
    file = np.load('mnist.npz')
    x_train = file.f.x_train
    y_train = file.f.y_train
    x_test = file.f.x_test
    y_test = file.f.y_test
    return x_train, y_train, x_test, y_test


def train():
    x_train, y_train, x_test, y_test = load_data()
    x_train = x_train.reshape(60000, 28, 28, 1) / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=30, batch_size=4096)

    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss:', loss)
    print('accuracy', accuracy)
    model.save('手写体训练模型/model1.h5')


def gray_trans(graph):
    gray = np.zeros((28, 28))
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            for each in graph[i][j]:
                gray[i][j] = gray[i][j] + each
            gray[i][j] = gray[i][j]/3   # 平均每个通道值转化为灰度图
    return gray



if __name__ == '__main__':

    # 图像化
    BLOCK = (20, 20)  # x，y
    AMOUNT = (28, 28)
    WRITE_SCREEN_SIZE = (BLOCK[0] * AMOUNT[0], BLOCK[1] * AMOUNT[1])  # x，y
    BlACK = (10, 10, 10)
    # WHITE = (255.0, 255.0, 255.0)
    WHITE = (246, 247, 241)
    GREY = (50, 50, 50)
    PENCIL = (15, 15, 10, 10)  # 方形笔触外圈宽，内圈宽

    gray_graph = [[[255, 255, 255] for i in range(AMOUNT[0])] for j in range(AMOUNT[1])]

    model = load_model('hand write recognize/手写体训练模型/model_4000.h5')

    pygame.init()
    screen = pygame.display.set_mode((WRITE_SCREEN_SIZE[0] + 1000, WRITE_SCREEN_SIZE[1]))
    big_font = pygame.font.Font('hand write recognize/手写体训练模型/ziti.ttf', 70)

    small_font = pygame.font.Font('hand write recognize/手写体训练模型/ziti.ttf', 30)

    screen.fill(WHITE)

    for row in range(AMOUNT[0] - 1):  # 横线
        pygame.draw.line(screen, color=BlACK, start_pos=(0, BLOCK[0] * (row + 1)),
                         end_pos=(WRITE_SCREEN_SIZE[0], BLOCK[0] * (row + 1)))
    for row in range(AMOUNT[1]):  # 竖线
        pygame.draw.line(screen, color=BlACK, start_pos=(BLOCK[0] * (row + 1), 0),
                         end_pos=(BLOCK[0] * (row + 1), WRITE_SCREEN_SIZE[1]))
    for number in range(10):
        pygame.draw.line(screen, BlACK,
                         (WRITE_SCREEN_SIZE[0] + 150 + 80*(number), BLOCK[1] * AMOUNT[1] - 60),
                         (WRITE_SCREEN_SIZE[0] + 150 + 80*(number), BLOCK[1] * AMOUNT[1] - 200))
        pygame.draw.line(screen, BlACK,
                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))
        pygame.draw.line(screen, BlACK,
                         (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60))
        pygame.draw.line(screen, BlACK,
                         (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200),
                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))

    # 显示文字部分
    screen.blit(big_font.render('None', True, BlACK), (1300, 100))
    for n in range(10):
        screen.blit(small_font.render(str(n), True, BlACK), (WRITE_SCREEN_SIZE[0] + 155 + 80 * n, BLOCK[1] * AMOUNT[1] - 55))

    pygame.display.flip()
    flag = False  # 检测鼠标是否按下的标记
    # 游戏循环
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                flag = True
            elif event.type == pygame.MOUSEBUTTONUP:
                flag = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    gray_graph = [[[255, 255, 255] for i in range(AMOUNT[0])] for j in range(AMOUNT[1])]

                    screen.fill(WHITE)
                    for row in range(AMOUNT[0] - 1):  # 横线
                        pygame.draw.line(screen, color=BlACK, start_pos=(0, BLOCK[0] * (row + 1)),
                                         end_pos=(WRITE_SCREEN_SIZE[0], BLOCK[0] * (row + 1)))
                    for row in range(AMOUNT[1]):  # 竖线
                        pygame.draw.line(screen, color=BlACK, start_pos=(BLOCK[0] * (row + 1), 0),
                                         end_pos=(BLOCK[0] * (row + 1), WRITE_SCREEN_SIZE[1]))
                    for number in range(10):
                        pygame.draw.line(screen, BlACK,
                                         (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                                         (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))
                        pygame.draw.line(screen, BlACK,
                                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))
                        pygame.draw.line(screen, BlACK,
                                         (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60))
                        pygame.draw.line(screen, BlACK,
                                         (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200),
                                         (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))
                        # 显示文字部分
                        screen.blit(big_font.render('None', True, BlACK), (1000, 100))
                        for n in range(10):
                            screen.blit(small_font.render(str(n), True, BlACK),
                                        (WRITE_SCREEN_SIZE[0] + 155 + 80 * n, BLOCK[1] * AMOUNT[1] - 55))

                if event.key == pygame.K_s:
                    np.save('data.npy', gray)
            if pygame.mouse.get_pressed(num_buttons=3) and flag:
                pos = pygame.mouse.get_pos()
                point = pencil_point(pos, PENCIL[:2])
                for each in point:
                    if 0 <= each[0] <= WRITE_SCREEN_SIZE[0] and 0 <= each[1] <= WRITE_SCREEN_SIZE[1]:
                        for n in range(4):  # 笔触的四角
                            for channel in range(3):
                                gray_graph[int(point[n][0] // BLOCK[0])][int(point[n][1] // BLOCK[1])][channel] -= GREY[
                                    channel]
                                if gray_graph[int(point[n][0] // BLOCK[0])][int(point[n][1] // BLOCK[1])][channel] >= 10:
                                    pass
                                else:
                                    gray_graph[int(point[n][0] // BLOCK[0])][int(point[n][1] // BLOCK[1])][channel] = 10
                            pygame.draw.rect(screen, gray_graph[int(point[n][0] // BLOCK[0])][int(point[n][1] // BLOCK[1])],
                                             (((point[n][0] // BLOCK[0]) * BLOCK[0], (point[n][1] // BLOCK[1]) * BLOCK[1]),
                                              BLOCK))
                        pygame.draw.rect(screen, BlACK,
                                         (((pos[0] // BLOCK[0]) * BLOCK[0], (pos[1] // BLOCK[1]) * BLOCK[1]), BLOCK))
                    else:
                        print('out of range!!')

                gray = gray_trans(gray_graph)
                gray = gray.T
                gray = gray.reshape(1, 28, 28, 1) / 255.0

                possibility = model.predict(gray)
                list = possibility[0].tolist()
                maximum = max(list)
                maxi = list.index(maximum)

                bar_length = possible_bar(possibility)

                pygame.draw.rect(screen, WHITE, (600, 0, 900, 505))  # 覆盖
                for number in range(10):
                    pygame.draw.line(screen, BlACK,
                                     (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                                     (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))
                    pygame.draw.line(screen, BlACK,
                                     (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                                     (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))
                    pygame.draw.line(screen, BlACK,
                                     (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60),
                                     (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 60))
                    pygame.draw.line(screen, BlACK,
                                     (WRITE_SCREEN_SIZE[0] + 150 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200),
                                     (WRITE_SCREEN_SIZE[0] + 180 + 80 * (number), BLOCK[1] * AMOUNT[1] - 200))

                for n in range(10):
                    pygame.draw.rect(screen, BlACK,
                                     (WRITE_SCREEN_SIZE[0] + 150 + 80 * n, BLOCK[1] * AMOUNT[1] - 60 - bar_length[n],
                                      30, bar_length[n] + 1)
                                     )
                # 显示文字部分
                screen.blit(big_font.render(str(maxi), True, BlACK), (1300, 100))

        # gray = gray_trans(gray_graph)
        # gray = gray.reshape(1, 28, 28, 1)/255.0
        # possibility = model.predict(gray)
        # bar_length = possible_bar(possibility)
        #
        # for n in range(10):
        #     pygame.draw.rect(screen, BlACK,
        #                      (WRITE_SCREEN_SIZE[0] + 150 + 80 * n, BLOCK[1] * AMOUNT[1] - 60 - bar_length[n],
        #                       30, bar_length[n] + 1)
        #                      )


        pygame.display.update()

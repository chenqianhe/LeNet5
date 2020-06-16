#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt 
import os


# In[16]:


BUF_SIZE = 512
BATCH_SIZE = 128

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(),
                          buf_size = BUF_SIZE),
batch_size = BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.test(),
                          buf_size = BUF_SIZE),
batch_size = BATCH_SIZE)

train_data = paddle.dataset.mnist.train()
sampledata = next(train_data())
print(sampledata)


# In[17]:


def convolutional_neural_network():

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')

    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act='relu'
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act='relu'
    )
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')

    return prediction


# In[18]:


def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = convolutional_neural_network()

    cost = fluid.layers.cross_entropy(input=predict, label=label)
    
    avg_cost = fluid.layers.mean(cost)

    acc = fluid.layers.accuracy(input=predict, label=label)

    return predict, [avg_cost, acc]
    


# In[19]:


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=1e-3)


# In[20]:


def event_handler(pass_id, batch_id, cost):
    print("Pass %d, Batch %d, Cost %f" % (pass_id, batch_id, cost))


# In[21]:


from paddle.utils.plot import Ploter

train_prompt = "Train cost"
test_prompt = "Test cost"
cost_ploter = Ploter(train_prompt, test_prompt)

def event_handler_plot(ploter_title, step, cost):
    cost_ploter.append(ploter_title, step, cost)
    cost_ploter.plot()


# In[22]:


use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace

prediction, [avg_loss, acc] = train_program()

img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')

label = fluid.layers.data(name='label', shape=[1], dtype='int64')

feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

optimizer = optimizer_program()
opts = optimizer.minimize(avg_loss)


# In[23]:


PASS_NUM = 10
epoch = [epoch_id for epoch_id in range(PASS_NUM)]

save_dirname = "recognize_digits.inference.model"


# In[24]:


def train_test(train_test_program, train_test_feed, train_test_reader):
    acc_set = []

    avg_loss_set = []

    for test_data in train_test_reader():
        acc_np, avg_loss_np = exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=[acc, avg_loss]
        )
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))

    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


# In[25]:


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# In[26]:


main_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)


# In[27]:


lists = []
step = 0
for epoch_id in epoch:
    for step_id, data in enumerate(train_reader()):
        metrics = exe.run(main_program,
                        feed=feeder.feed(data),
                        fetch_list=[avg_loss, acc])
        if step % 100 == 0:
            event_handler(step, epoch_id, metrics[0])

            # event_handler_plot(train_prompt, step, metrics[0])

        step += 1

    print("finish")

    avg_loss_val, acc_val = train_test(train_test_program=test_program, 
                                        train_test_reader=test_reader,
                                        train_test_feed=feeder)
    
    print("Test with Epoch %d, avg_cost: %s, acc: %s" % (epoch_id, avg_loss_val, acc_val))
    # event_handler_plot(test_prompt, step, metrics[0])

    lists.append((epoch_id, avg_loss_val, acc_val))

    if save_dirname is not None:
        fluid.io.save_inference_model(save_dirname,
                                    ["img"], [prediction], exe,
                                    model_filename=None,
                                    params_filename=None)



# In[28]:


best = sorted(lists, key=lambda list:float(list[1]))[0]
print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
print('The classification accuracy is %.2f%%' % (float(best[2])*100))


# In[ ]:





# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

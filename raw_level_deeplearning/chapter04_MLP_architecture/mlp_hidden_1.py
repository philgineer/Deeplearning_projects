import numpy as np


# hyperparams
RND_MEAN = 0.0
RND_STD = 0.1
LEARNING_RATE = 0.01


def init_model():
    global pm_output, pm_hidden, input_cnt, output_cnt, hidden_cnt
    pm_hidden = alloc_param_pair([input_cnt, hidden_cnt])
    pm_output = alloc_param_pair([hidden_cnt, output_cnt])


def alloc_param_pair(shape):
    weight = np.random.normal(RND_MEAN, RND_STD, shape)
    bias = np.zeros(shape[-1])

    return {'w': weight, 'b': bias}


def forward_neuralnet(x):
    global pm_hidden, pm_output

    hidden = relu(np.matmul(x, pm_hidden['w']) +pm_hidden['b'])
    output = np.matmul(hidden, pm_output['w']) + pm_output['b']

    return output, [x, hidden]


def relu(x):
    return np.maximum(x, 0)


def backprop_neuralnet(G_output, aux):
    global pm_hidden, pm_output

    x, hidden = aux

    ### 1. output -> hidden ###
    g_output_w_out = hidden.transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)

    # pm_output['w'] 값이 업데이트 되기 전에 미리 G_hidden 계산
    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out

    ### 2. hidden -> input ###
    G_hidden = G_hidden * relu_derv(hidden)

    g_hidden_w_hid = x.transpose()
    G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
    G_b_hid = np.sum(G_hidden, axis=0)

    pm_hidden['w'] -= LEARNING_RATE * G_w_hid
    pm_hidden['b'] -= LEARNING_RATE * G_b_hid


def relu_derv(y):
    return np.sign(y)

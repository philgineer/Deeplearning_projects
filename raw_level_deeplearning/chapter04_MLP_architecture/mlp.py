import numpy as np

global hidden_config

def init_model():
    print("MLP with {} hidden layers started learning.".format(len(hidden_config)))
    init_model_hiddens()


def set_hidden(info):
    global hidden_cnt, hidden_config
    if isinstance(info, int):
        hidden_cnt = info
        hidden_config = None
    else:
        hidden_config = info


def init_model_hiddens():
    global pm_output, pm_hiddens, input_cnt, output_cnt, hidden_config

    pm_hiddens = []
    prev_cnt = input_cnt

    for hidden_cnt in hidden_config:
        pm_hiddens.append(alloc_param_pair([prev_cnt, hidden_cnt]))
        prev_cnt = hidden_cnt

    pm_output = alloc_param_pair([prev_cnt, output_cnt])


def alloc_param_pair(shape):
    weight = np.random.normal(RND_MEAN, RND_STD, shape)
    bias = np.zeros(shape[-1])

    return {'w': weight, 'b': bias}


def forward_neuralnet_hiddens(x):
    global pm_hiddens, pm_output

    hidden = x
    hiddens = [x]

    for pm_hidden in pm_hiddens:
        hidden = relu(np.matmul(x, pm_hidden['w']) +pm_hidden['b'])
        hiddens.append(hidden)

    output = np.matmul(hidden, pm_output['w']) + pm_output['b']

    return output, hiddens


def relu(x):
    return np.maximum(x, 0)


def backprop_neuralnet_hiddens(G_output, aux):
    global pm_hiddens, pm_output

    hiddens = aux

    ### 1. output -> hidden ###
    g_output_w_out = hiddens[-1].transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)

    # pm_output['w'] 값이 업데이트 되기 전에 미리 G_hidden 계산
    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out

    ### 2. last hidden -> ... -> first hidden -> input ###
    for n in reversed(range(len(pm_hiddens))):
        G_hidden = G_hidden * relu_derv(hiddens[n+1])
        
        g_hidden_w_hid = hiddens[n].transpose()
        G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
        G_b_hid = np.sum(G_hidden, axis=0)
        
        # pm_hiddens[n]['w'] 값 업데이트 전에 G_hidden 업데이트
        g_hidden_hidden = pm_hiddens[n]['w'].transpose()
        G_hidden = np.matmul(G_hidden, g_hidden_hidden)

        pm_hiddens[n]['w'] -= LEARNING_RATE * G_w_hid
        pm_hiddens[n]['b'] -= LEARNING_RATE * G_b_hid


def relu_derv(y):
    return np.sign(y)

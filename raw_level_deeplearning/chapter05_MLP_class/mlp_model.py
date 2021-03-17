from matutil import *

np.random.seed(1234)
def randomize():
    np.random.seed(int(time.time()))


class Model(object):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.is_training = False
        if not hasattr(self, 'rand_std'):
            self.rand_std = 0.030

    def __str__(self):
        return '{}/{}'.format(self.name, self.dataset)

    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001,
                report=0, show_cnt=3):
        self.train(epoch_count, batch_size, learning_rate, report)
        self.test()
        if show_cnt > 0:
            self.visualize(show_cnt)


class MlpModel(Model):
    def __init__(self, name, dataset, hconfigs):
        super(MlpModel, self).__init__(name, dataset)
        self.init_parameters(hconfigs)

def mlp_init_parameters(self, hconfigs):
    self.hconfigs = hconfigs
    self.pm_hiddens = []

    prev_shape = self.dataset.input_shape

    for hconfig in hconfigs:
        pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
        self.pm_hiddens.append(pm_hidden)

    output_cnt = int(np.prod(self.dataset.output_shape))
    self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)

def mlp_alloc_layer_param(self, input_shape, hconfig):
    input_cnt = np.prod(input_shape)
    output_cnt = hconfig

    weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

    return {'w': weight, 'b': bias}, output_cnt

def mlp_alloc_param_pair(self, shape):
    weight = np.random.normal(0, self.rand_std, shape)
    bias = np.zeros([shape[-1]])
    return weight, bias

MlpModel.init_parameters = mlp_init_parameters
MlpModel.alloc_layer_param = mlp_alloc_layer_param
MlpModel.alloc_param_pair = mlp_alloc_param_pair

def mlp_model_train(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
    self.learning_rate = learning_rate

    batch_count = int(self.dataset.train_count / batch_size)
    time1 = time2 = int(time.time())
    if report != 0:
        print('Model {} train started:'.format(self.name))

    for epoch in range(epoch_count):
        costs = []
        accs = []
        self.dataset.shuffle_train_data(batch_size * batch_count)
        for n in range(batch_count):
            trX, trY = self.dataset.get_train_data(batch_size, n)
            cost, acc = self.train_step(trX, trY)
            costs.append(cost)
            accs.append(acc)

        if report > 0 and (epoch + 1) % report == 0:
            vaX, vaY = self.dataset.get_validate_data(100)
            acc = self.eval_accuracy(vaX, vaY)
            time3 = int(time.time())
            tm1, tm2 = time3 - time2, time3 - time1
            self.dataset.train_prt_result(epoch + 1, costs, accs, acc, tm1, tm2)
            time2 = time3

        tm_total = int(time.time()) - time1
        print('Model {} train ended in {} secs:'.format(self.name, tm_total))

MlpModel.train = mlp_model_train

def mlp_model_test(self):
    teX, teY = self.dataset.get_test_data()
    time1 = int(time.time())
    acc = self.eval_accuracy(teX, teY)
    time2 = int(time.time())
    self.dataset.test_prt_result(self.name, acc, time2 - time1)

MlpModel.test = mlp_model_test

def mlp_model_visualize(self, num):
    print('Model {} Visualization'.format(self.name))
    deX, deY = self.dataset.get_visualize_data(num)
    est = self.get_estimate(deX)
    self.dataset.visualize(deX, est, deY)

MlpModel.visualize = mlp_model_visualize

def mlp_train_step(self, x, y):
    self.is_training = True

    output, aux_nn = self.forward_neuralnet(x)
    loss, aux_pp = self.forward_postproc(output, y)
    accuracy = self.eval_accuracy(x, y, output)

    G_loss = 1.0
    G_output = self.backprop_postproc(G_loss, aux_pp)
    self.backprop_neuralnet(G_output, aux_nn)

    self.is_training = False

    return loss, accuracy

MlpModel.train_step = mlp_train_step

def mlp_forward_neuralnet(self, x):
    hidden = x
    aux_layers = []

    for n, hconfig in enumerate(self.hconfigs):
        hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])
        aux_layers.append(aux)

    output, aux_out = self.forward_layer(hidden, None, self.pm_output)

    return output, [aux_out, aux_layers]


def mlp_backprop_neuralnet(self, G_output, aux):
    aux_out, aux_layers = aux

    G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)

    for n in reversed(range(len(self.hconfigs))):
        hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
        G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

    return G_hidden

MlpModel.forward_neuralnet = mlp_forward_neuralnet
MlpModel.backprop_neuralnet = mlp_backprop_neuralnet

def mlp_forward_layer(self, x, hconfig, pm):
    y = np.matmul(x, pm['w']) + pm['b']
    if hconfig is not None: y = relu(y)
    return y, [x, y]

def mlp_backprop_layer(self, G_y, hconfig, pm, aux):
    x, y = aux
    if hconfig is not None: G_y = relu_derv(y) * G_y

    g_y_weight = x.transpose()
    g_y_input = pm['w'].transpose()

    G_weight = np.matmul(g_y_weight, G_y)
    G_bias = np.sum(G_y, axis=0)
    G_input = np.matmul(G_y, g_y_input)

    pm['w'] -= self.learning_rate * G_weight
    pm['b'] -= self.learning_rate * G_bias

    return G_input

MlpModel.forward_layer = mlp_forward_layer
MlpModel.backprop_layer = mlp_backprop_layer

def mlp_forward_postproc(self, output, y):
    loss, aux_loss = self.dataset.forward_postproc(output, y)
    extra, aux_extra = self.forward_extra_cost(y)
    return loss + extra, [aux_loss, aux_extra]

def mlp_forward_extra_cost(self, y):
    # regularization cost
    return 0, None

MlpModel.forward_postproc = mlp_forward_postproc
MlpModel.forward_extra_cost = mlp_forward_extra_cost

def mlp_backprop_postproc(self, G_loss, aux):
    aux_loss, aux_extra = aux
    self.backprop_extra_cost(G_loss, aux_extra)
    G_output = self.dataset.backprop_postproc(G_loss, aux_loss)
    return G_output

def mlp_backprop_extra_cost(self, G_loss, aux):
    pass

MlpModel.backprop_postproc = mlp_backprop_postproc
MlpModel.backprop_extra_cost = mlp_backprop_extra_cost

def mlp_eval_accuracy(self, x, y, output=None):
    if output is None:
        output, _ = self.forward_neuralnet(x)
    accuracy = self.dataset.eval_accuracy(x, y, output)
    return accuracy

MlpModel.eval_accuracy = mlp_eval_accuracy

def mlp_get_estimate(self, x):
    output, _ = self.forward_neuralnet(x)
    estimate = self.dataset.get_estimate(output)
    return estimate

MlpModel.get_estimate = mlp_get_estimate

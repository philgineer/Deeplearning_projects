from dataset import *

class AbaloneDataset(Dataset):
    def __init__(self):
        super(AbaloneDataset, self).__init__('abalone', 'regression')

        rows, _ = load_csv('../../../../data/raw_level/abalone.csv')

        xs = np.zeros([len(rows), 10])
        ys = np.zeros([len(rows), 1])

        for n, row in enumerate(rows):
            if row[0] == 'I': xs[n, 0] = 1
            if row[0] == 'M': xs[n, 1] = 1
            if row[0] == 'F': xs[n, 2] = 1
            xs[n, 3:] = row[1:-1]
            ys[n, :] = row[-1:]

        self.shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = vector_to_str(x, '%4.2f')
            print('{} => predict {:4.1f} : ground_truth {:4.1f}'.format(xstr, est[0], ans[0]))


class PulsarDataset(Dataset):
    def __init__(self):
        super(PulsarDataset, self).__init__('pulsar', 'binary')

        rows, _ = load_csv('../../../../data/raw_level/pulsar_stars.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:, :-1], data[:, -1:], 0.8)
        self.target_names = ['star', 'pulsar']

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = vector_to_str(x, '%5.1f', 3)
            estr = self.target_names[int(round(est[0]))]
            astr = self.target_names[int(round(ans[0]))]
            rstr = '0'
            if estr != astr: rstr = 'X'
            print('{} => predict {}(prob {:4.2f}) : ground_truth {} => {}'.
                    format(xstr, estr, est[0], astr, rstr))


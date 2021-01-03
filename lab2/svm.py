import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

N_FEATURES = 123
N_EPOCH = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 512
C = 100

def preprocess_X(X):
    return np.hstack((np.asarray(X.todense()), np.full((X.shape[0], 1), 1.0)))

def acc(pred, y):
    return np.count_nonzero((pred > 0) == (y > 0)) / pred.shape[0]

def main():
    train_X, train_y = sklearn.datasets.load_svmlight_file('data/a9a', n_features=N_FEATURES)
    test_X, test_y = sklearn.datasets.load_svmlight_file('data/a9a.t', n_features=N_FEATURES)

    train_X = preprocess_X(train_X)
    test_X = preprocess_X(test_X)

    weight = np.zeros((N_FEATURES + 1,))
    def loss_fn(pred, y):
        return (weight @ weight) / 2 + C * np.mean(np.maximum(0, 1 - y * pred))

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    n_iter = train_X.shape[0] // BATCH_SIZE
    for epoch in range(N_EPOCH):
        idx = np.random.permutation(train_X.shape[0])
        total_loss = 0
        total_acc = 0
        for i in range(n_iter):
            iter_idx = idx[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
            X = train_X[iter_idx]
            y = train_y[iter_idx]
            pred = X @ weight
            grad = weight + C * np.mean(np.where(y * pred <= 1, -y * X.T, 0), axis=1)

            weight -= LEARNING_RATE * grad
            total_loss += loss_fn(pred, y)
            total_acc += acc(pred, y)
        total_loss /= n_iter
        total_acc /= n_iter
        train_loss.append(total_loss)
        train_acc.append(total_acc)

        test_pred = test_X @ weight
        test_loss.append(loss_fn(test_pred, test_y))
        test_acc.append(acc(test_pred, test_y))

    best = np.argmax(test_acc)
    print('Train loss', train_loss[best])
    print('Test loss', test_loss[best])
    print('Accuracy', test_acc[best])

    x = range(1, N_EPOCH + 1)
    plt.figure(figsize=(4, 2.5))
    plt.xlabel('#epoch')
    plt.ylabel('loss')
    plt.plot(x, train_loss, label='train')
    plt.plot(x, test_loss, label='test')
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig('report/figures/svm/loss.pdf')
    plt.close()

    plt.figure(figsize=(4, 2.5))
    plt.xlabel('#epoch')
    plt.ylabel('Accuracy (%)')
    plt.plot(x, [v * 100 for v in train_acc], label='train')
    plt.plot(x, [v * 100 for v in test_acc], label='test')
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig('report/figures/svm/acc.pdf')
    plt.close()

if __name__ == "__main__":
    main()

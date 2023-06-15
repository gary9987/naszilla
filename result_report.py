import pickle
import os
import numpy as np


if __name__ == '__main__':
    folder = 'results_output'
    algo = 'local_search'
    dataset = 'cifar100'
    val_acc_list = []
    test_acc_list = []
    for seed in range(10):
        with open(os.path.join(folder, algo, dataset, f'round_{seed}.pkl'), 'rb') as f:
            results = pickle.load(f)
            test_acc = results[2][0].tolist()
            val_acc = results[-1][0].tolist()
            #print(val_acc)
            val_acc_list.append(100. - val_acc[-2][-1])
            test_acc_list.append(100. - test_acc[-2][-1])
    
    print(val_acc_list)
    print(test_acc_list)
    print(np.mean(val_acc_list), np.std(val_acc_list))
    print(np.mean(test_acc_list), np.std(test_acc_list))

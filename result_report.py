import pickle
import os
import numpy as np


if __name__ == '__main__':
    folder = 'results_output'
    algo = 'bananas'
    dataset = 'nasbench_101/cifar10'
    val_acc_list = []
    test_acc_list = []
    for seed in range(10):
        with open(os.path.join(folder, algo, dataset, f'round_{seed}.pkl'), 'rb') as f:
            results = pickle.load(f)
            test_acc = results[2][0].tolist()
            val_acc = results[-1][0].tolist()
            now_min = val_acc[0][-1]
            for i in range(len(val_acc)):
                val_acc[i][-1] = min(val_acc[i][-1], now_min)
                now_min = min(now_min, val_acc[i][-1])
            now_min = test_acc[0][-1]
            for i in range(len(test_acc)):
                test_acc[i][-1] = min(test_acc[i][-1], now_min)
                now_min = min(now_min, test_acc[i][-1])

            print(val_acc[18])
            #print(test_acc)
            val_acc_list.append(100. - val_acc[18][-1])
            test_acc_list.append(100. - test_acc[18][-1])
            
    
    print(val_acc_list)
    print(test_acc_list)
    print(np.mean(val_acc_list), np.std(val_acc_list))
    print(np.mean(test_acc_list), np.std(test_acc_list))

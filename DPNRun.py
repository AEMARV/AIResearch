import os
import sys

global server_mode
from src.experiments import *

if __name__ == '__main__':

    desc = ""
    augment_rate = 0
    datasets = [(cifar10, 10)]
    alpha_list = [2]
    exp_list = []
    for trial in range(5):
        for lr in [0.01]:
            for batch_size in [128]:
                for alpha in alpha_list:
                    for numlayers in [12]:
                        for dataset, classnum in datasets:
                            for optimizer in [Joint_Probabilistic]:
                                for scale in [1]:
                                    for model in [DPN_SimpleCIFAR10,
                                                  ]:
                                        exp_desc = dict(trial=trial,
                                                        lr=lr,
                                                        l2=0,
                                                        l1=0,
                                                        filter_scale=scale,
                                                        momentum=0.0,
                                                        batchsz=batch_size,
                                                        alpha=alpha,
                                                        model=model,
                                                        num_layer=numlayers,
                                                        dataset=dataset,
                                                        classnum=classnum,
                                                        optimizer=optimizer)
                                    exp_list = exp_list + [exp_desc]

    worker_id = 1
    total_workers = 1
    if sys.argv.__len__() > 1:
        worker_id = int(sys.argv[1]) - 1
        total_workers = int(sys.argv[2])
        exp_list = exp_list[worker_id::total_workers]
        print("The script is running with batch sz " + str(alpha_list))

    print(exp_list.__len__())
    for exp_dict in exp_list:
        exp = Trainings_SimpleCIFAR10('DPN', description=desc)
        exp.generic_train(**exp_dict)
#

# exp.run()
# exp = VGG_PMAP_CIFAR10_Try(1)
# exp = Synthetic_PMaps(1)
# exp.run()

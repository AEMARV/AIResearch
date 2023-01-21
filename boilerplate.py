import os
import sys

global server_mode
from src.experiments import *

if __name__ == '__main__':

    # exp = NIN_Dropout(1)
    desc = '''Experimenting with Bottle Net. BottleNet aggregates the output across layers to 
     combat gradient vanishing. '''
    augment_rate = 0
    exp_list = []
    total_trials = ['max-min']
    datasets = [(cifar10, 10)]

    alpha_list = [4,2,1]
    l1_list = [0]
    l2_list = l1_list
    numlayers_list = [24]
    lr_list = [0.001]
    init_coef_list = [2]
    optimizer_list = [SGD]
    scale_list = [1]
    loss_list = [
                Conditional_Subset,
                Conditional_Intersection
                 # Joint_Cross,
                 # Joint_Intersection_Subset,
                 # Joint_Intersection_Indpt
    ]
    model_list = [BottleNet
                  # SimpleCIFAR10,
                  # resnet_cifar_nmnist
                  ]

    for trial in total_trials:
        for batch_size in [128]:
            for alpha in alpha_list:
                for numlayers in numlayers_list:
                    for dataset, classnum in datasets:
                        for scale in scale_list:
                            for l1 in l1_list:
                                for l2 in l2_list:
                                    for lr in lr_list:
                                        for init_coef in init_coef_list:
                                            for optimizer in optimizer_list:
                                                for loss in loss_list:
                                                    for model in model_list:
                                                        exp_desc = dict(trial=trial,
                                                                        lr=lr,
                                                                        l2=l2,
                                                                        l1=l1,
                                                                        init_coef=init_coef,
                                                                        filter_scale=scale,
                                                                        momentum=0.9,
                                                                        batchsz=batch_size,
                                                                        alpha=alpha,
                                                                        model=model,
                                                                        num_layer=numlayers,
                                                                        dataset=dataset,
                                                                        classnum=classnum,
                                                                        optimizer=optimizer,
                                                                        loss=loss)
                                                        exp_list = exp_list + [exp_desc]


    worker_id = 1
    total_workers = 1
    print("Experimenting on total of %d settings" % exp_list.__len__())
    if sys.argv.__len__() > 1:
        worker_id = int(sys.argv[1]) - 1
        total_workers = int(sys.argv[2])
        exp_list = exp_list[worker_id::total_workers]
    print("This worker is experimenting on total of %d settings" % exp_list.__len__())
    for exp_dict in exp_list:
        exp = Trainings_SimpleCIFAR10('BottleNet', description=desc, worker_id=worker_id)
        exp.generic_train(**exp_dict)
#


# exp.run()
# exp = VGG_PMAP_CIFAR10_Try(1)
# exp = Synthetic_PMaps(1)
# exp.run()


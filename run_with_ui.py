import os
import sys

global server_mode
from src.experiments import *
from torch.optim import Adam
from GUI import mainframecode
import sys
import multiprocessing as mp
from multiprocessing import Queue as Queue
from PyQt6.QtWidgets import QApplication, QWidget,QMainWindow
import sys
import pygames

class hyperparam():
    possible_types = ['num','choice']
    def __init__(self,type,state,possibilities):
        self.type = ['numerical', 'choice']
        self.state = None
        self.possiblities = []

class setup():

    def __init__(self, hyperparam_list):
        self.hyper_params = hyperparam_list

def launch_gui(queue:Queue,setup_obj):
    app = QApplication(sys.argv)

    # Create a Qt widget, which will be our window.
    main_window = QMainWindow()
    window = mainframecode.Ui_MainWindow(queue,setup_obj)
    window.setupUi(main_window,setup_obj)
    main_window.show()  # IMPORTANT!!
    app.exec()
if __name__ == '__main__':

    GUI_P = mp.Process(target=launch_gui,name='gui_process',args=(None,None))
    GUI_P.daemon = True
    GUI_P.start()
    GUI_P.join()



    # exp = NIN_Dropout(1)
    ''' Detect panc on NIH
	1. create

	'''
    ''' Detect Panc on Mayo'''
    ''' Train EBM on NIH and MAYO'''
    ''' TEST Phase'''
    desc = "This experiment is designed to test variation of alpha param and multiple loss functions" \
           "\nThe loss functions are as followed:\n" \
           "\n1. Cross Entropy Conditional/Joint\n" \
           "\nCross Entropy means that each sample affects the network independantly\n" \
           "\nCond/Joint reflect that the network is trained on input/label ( joint) or just the label (Cond) " \
           "" \
           "\n2. Probabilistic Conditional/Joint" \
           "\nIn Probabilistic Training each sample affects the gradient based on the probability of correctness." \
           "\n In other words, the marginalized probability, is optimized" \
           "" \
           "" \
           "In addition to the previous objecrtive function a v2 for Joint version is experimented:\n" \
           "where in hyper normalization as opposed to the original, the original input is not used."



    augment_rate = 0
    exp_list = []
    total_trials = [2]
    datasets = [(nih_pancreas2d, 2)]

    alpha_list = [1]
    l1_list = [0]
    l2_list = l1_list
    numlayers_list = [24]
    lr_list = [0.001]
    init_coef_list = [3]
    optimizer_list = [SGD]
    loss_list = [Cross_Seg_Balanced_NOEBM]
    model_list = [Panc_Seg_Bottled_FullScale_2D]
    scale_list = [1]


    for trial in total_trials:
        for batch_size in [1]:
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
    # for exp_dict in exp_list:
    #     exp = Trainings_SimpleCIFAR10('Full_Scale', description=desc, worker_id=worker_id)
    #     exp.generic_train(**exp_dict)
#


# exp.run()
# exp = VGG_PMAP_CIFAR10_Try(1)
# exp = Synthetic_PMaps(1)
# exp.run()


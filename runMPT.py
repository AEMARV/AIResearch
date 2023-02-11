import os
import sys
global server_mode
from src.experiments import *
import src.zoo.pop as popmodels
if __name__ == '__main__':


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
	total_trials= 1
	datasets= [(cifar10,10),(cifar100,100), (mnist, 10)]

	alpha_list=[1,2,3,4,5,6,12,24]
	l1_list = [0,2e-5,1e-4]
	l2_list = l1_list
	numlayers_list= [12]
	lr_list = [0.1, 0.01, 0.001,0.0001]
	init_coef_list = [1]
	optimizer_list = [SGD
					  # Probabilistic SGD
					  ]
	loss_list = [Conditional_Cross,
				 Joint_Cross,
				 Joint_Intersection_Indpt]
	model_list = [
				popmodels.ResNet50,
				popmodels.DenseNet121,
				popmodels.VGG,
				popmodels.densenet_cifar,
				popmodels.SimpleDLA,
				popmodels.DLA,
				popmodels.DPN92,
				BottleNet
				  ]


	for trial in range(total_trials):
		for batch_size in [64]:
			for alpha in alpha_list:
				for numlayers in numlayers_list:
					for dataset,classnum in datasets:
						for scale in [1]:
							for l1 in l1_list:
								for l2 in l2_list:
									for lr in lr_list:
										for init_coef in init_coef_list:
											for optimizer in optimizer_list :
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
														exp_list = exp_list +[exp_desc]

	alpha_list = [1]
	l1_list = [2e-5,4e-5,8e-5,16e-5]
	l2_list = [0]


	worker_id= 1
	total_workers=1
	print("Experimenting on total of %d settings"%exp_list.__len__())
	if sys.argv.__len__()>1:
		worker_id= int(sys.argv[1])-1
		total_workers= int(sys.argv[2])
		exp_list=  exp_list[worker_id::total_workers]
	print("This worker is experimenting on total of %d settings"%exp_list.__len__())
	for exp_dict in exp_list:
		exp = Trainings_SimpleCIFAR10('MassMPT2023', description=desc,worker_id=worker_id)
		exp.generic_train(**exp_dict)
	#





	# exp.run()
	# exp = VGG_PMAP_CIFAR10_Try(1)
	# exp = Synthetic_PMaps(1)
	# exp.run()


from src.losses import *
from src.zoo.models import *
# from src.experiment import Experiment_
from src.data import *
from matplotlib import pyplot as plt
from src.utils import *
from torch.utils.tensorboard import SummaryWriter
import shutil
import torch
from src.optimizers import *


class Experiment:
    def __init__(self, device='cuda:0', worker_id=1):
        '''
		Base Class for experiments. The class has the following properties
		device,
		tensorboard summarywriter
		a dictionary containing the results
		:param device: preset to cuda0
		'''
        self.device = device
        self.writer = None
        self.result_dict = {}
        self.path = os.path.join('.', 'Results')
        temp_code_path = os.path.join('.', 'temp', 'code_%d' % worker_id)
        self.temp_code_path = temp_code_path
        if os.path.exists(self.temp_code_path):
            shutil.rmtree(self.temp_code_path)
        os.makedirs(self.temp_code_path)
        copy_code(self.temp_code_path)

    def add_epoch_res_dict(self, result_dict, resdict: dict, epoch, write):
        for i, key in enumerate(resdict.keys()):
            if key in result_dict.keys():
                result_dict[key].append(resdict[key])
            else:
                result_dict[key] = [resdict[key]]
            if write:
                if self.writer is None:
                    print('creating summary file at ', self.path)
                    path = os.path.abspath(self.path)
                    self.writer = SummaryWriter(path)
                self.writer.add_scalar(key, resdict[key], epoch)

        return result_dict

    def print(self, string, end='\n'):
        '''
		prints to the stdout and the log file, where path is defined in the object
		:param string:
		:param end:
		:return:
		'''
        path = self.path
        log_file = open(os.path.join(path, 'log.txt'), "a")
        print(str(string), end=end)
        log_file.write(str(string) + end)
        log_file.close()

    def print_dict(self, res_dict: dict, prefix='', postfix='\n'):
        string = prefix
        for key, val in res_dict.items():
            string = string + ' ' + key + ': ' + '%.4f' % val
        string = string
        self.print(string, end=postfix)

    def train_epoch(self, model,
                    loss: Loss,
                    optimizer,
                    trainloader,
                    testloader,
                    ):
        ''' The function returns a dict of scalars, the statistics of the epoch
		def train_epoch(self, model,
					optimizer:Optimizer,
					trainloader,
					testloader,
					prefix_text,
					path) -> dict:

		'''
        prefixprint = "Epoch " + (optimizer.trained_epochs + 1).__str__() + "| "
        plt.show()
        totalsamples = 0
        ISNAN = False
        avg_train_result = None
        avg_val_result = None

        for batch_n, data in enumerate(trainloader):

            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            this_samp_num = inputs.shape[0]
            model.train()
            temp_result = loss.calc_grad(model, inputs, labels)
            diverged = False

            temp_result = {str(key) + '_train': val for key, val in temp_result.items()}
            for key in temp_result.keys():
                diverged = True if hasnan(temp_result[key]) else diverged
            optimizer.step(model)
            if avg_train_result is None:
                avg_train_result = temp_result
            else:
                avg_train_result = dict_lambda(avg_train_result, temp_result,
                                               f=lambda x, y: (x * totalsamples + y * this_samp_num) / (
                                                       totalsamples + this_samp_num))
            ''' Print Output'''
            print("", end='\r')
            self.print_dict(avg_train_result, prefix='Train: ' + prefixprint + '| ' + str(batch_n) + ':', postfix=" ")
            totalsamples = totalsamples + inputs.shape[0]
            if diverged:
                ISNAN = True
                break
        print("", end='\r')

        model.eval()
        totalsamples = 0
        for batch_n, data in enumerate(testloader):
            with torch.set_grad_enabled(False):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                this_samp_num = inputs.shape[0]
                temp_result = loss.calc_grad(model, inputs, labels)
                temp_result = {str(key) + '_val  ': val for key, val in temp_result.items()}
                if avg_val_result is None:
                    avg_val_result = temp_result
                else:
                    avg_val_result = dict_lambda(avg_val_result, temp_result,
                                                 f=lambda x, y: (x * totalsamples + y * this_samp_num) / (
                                                         totalsamples + this_samp_num))

                totalsamples = totalsamples + this_samp_num

            if ISNAN:
                break

        self.print_dict(avg_train_result, prefix='Train: ' + prefixprint)
        self.print_dict(avg_val_result, prefix='Test : ' + prefixprint)
        print('')
        avg_total_result = avg_train_result
        avg_total_result.update(avg_val_result)

        return avg_total_result, ISNAN

    def train_all(self, model,
                  loss,
                  optimizer,
                  trainldr,
                  testldr,
                  path: str,
                  setup: dict,
                  save_result,
                  epochnum, device='cuda:0'):
        """
		:param model:
		a module object
		:param optimizer:
		an optimizer object
		:param trainldr:
		data loader
		:param testldr:
		data loader
		:param path:
		string
		:param save_result:
		bool True/False
		:param epochnum:
		:param device:
		default: cuda:0
		:return:
		"""
        model_path = os.path.join(path, 'model.pth')
        loss_path = os.path.join(path, 'loss.pth')
        optimizer_path = os.path.join(path, "optimizer.pth")
        result_path = os.path.join(path, 'result.dict')
        if (os.path.exists(model_path)):
            model = torch.load(model_path)
            loss = torch.load(loss_path)
            optimizer = torch.load(optimizer_path)
        if os.path.exists(result_path):
            result_dict = torch.load(result_path)
        else:
            result_dict = {'setup': setup, 'scalars': dict()}
        model.to(device)
        loss.to(device)
        self.path = path
        print(path)
        ISNAN = False
        torch.save(model, model_path)
        torch.save(loss, loss_path)

        for epoch in range(optimizer.trained_epochs + 1, epochnum):
            if not ISNAN:
                epochres, ISNAN = self.train_epoch(model, loss,optimizer, trainldr, testldr)
            optimizer.inc_epoch()
            result_dict['scalars'] = self.add_epoch_res_dict(result_dict['scalars'], epochres, epoch, save_result)
            # print(result_dict['scalars'])
            if save_result:
                if not ISNAN:
                    torch.save(model, model_path)
                torch.save(optimizer, optimizer_path)
                torch.save(loss, loss_path)
                torch.save(result_dict, os.path.join(path, 'result.dict'))
        model.to(torch.device('cpu'))
        loss.to(torch.device('cpu'))
        return result_dict


class Trainings_SimpleCIFAR10(Experiment):
    def __init__(self, res_folder_name, description='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_num = 150
        self.folder_name = res_folder_name
        self.description = description

        self.path = os.path.join('.', 'Results', self.folder_name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.desc_path = os.path.join(self.path, 'description.txt')
        if os.path.exists(self.desc_path):
            os.remove(self.desc_path)
        dsc_file = open(self.desc_path, mode='x')
        dsc_file.write(self.description)
        dsc_file.close()

    def reset_path(self):
        self.writer = None
        self.result_dict = {}
        self.path = os.path.join('.', 'Results', self.folder_name)

    def generic_train(self, trial=1, alpha=2,
                      augment_rate=0,
                      batchsz=128,
                      lr=0.01,
                      filter_scale=1,
                      momentum=0.9,
                      num_layer=12,
                      l2=0,
                      l1=0,
                      init_coef=1,
                      model=SimpleCIFAR10,
                      loss=Joint_Subset,
                      optimizer=SGD,
                      dataset=cifar10, name="",
                      classnum=-1):
        self.reset_path()
        model = model(layers=num_layer, classnum=classnum, filter_scale=filter_scale,
                      init_coef=init_coef)  # type: torch.nn.Module
        optimizer = optimizer(lr=lr, momentum=momentum, l1=l1, l2=l2)
        loss = loss(alpha, classnum, augment_rate, lr=lr, momentum=momentum, l1=l1, l2=l2)
        train_dataldr, test_dataldr = dataset(batchsz)
        ## Create Results folder
        setup = dict(model_name=model.__class__.__name__,
                     optimizer=optimizer.__class__.__name__,
                     loss=loss.__class__.__name__,
                     filter_scale=filter_scale,
                     init_coef=init_coef,
                     momentum=momentum,
                     alpha=alpha,
                     batchsz=batchsz,
                     lr=lr,
                     layers=num_layer,
                     trial=trial
                     )

        name = name + dict_filename(setup)
        self.path = os.path.join(self.path, dataset.__name__, name)
        if os.path.exists(self.path):
            import warnings
            warnings.warn("experiment exists and is going to be overwritten")
        os.makedirs(self.path, exist_ok=True)
        code_path = os.path.join(self.path, 'Code')
        os.makedirs(code_path, exist_ok=True)
        # Create setup description file
        setup_desc_path = os.path.join(self.path, 'setup.txt')
        if os.path.exists(setup_desc_path):
            os.remove(setup_desc_path)
        setup_txt = dict_to_str(setup)
        print(setup_txt)
        dsc_file = open(setup_desc_path, mode='x')
        dsc_file.write("Setup : \n\n" + setup_txt)
        dsc_file.close()
        copy_code(code_path, rootpath=self.temp_code_path)
        result = self.train_all(model,
                                loss,
                                optimizer,
                                train_dataldr,
                                test_dataldr,
                                self.path,
                                setup,
                                True,
                                self.epoch_num,

                                )
        shutil.rmtree(self.temp_code_path)
        return result
# result.update({"setup": setup})
# torch.save(result, os.path.join(self.path, 'result_dict.results'))

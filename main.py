import importlib
import datetime
import argparse
import time
import os
# import ipdb
from tqdm import tqdm

import torch
from torch.autograd import Variable
import parser as file_parser
from metrics.metrics import confusion_matrix
from utils import misc_utils
from main_multi_task import life_experience_iid, eval_iid_tasks

# eval_class_tasks(model, tasks, args) : returns lists of avg losses after passing thru model
# eval_tasks(model, tasks, args) : ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# life_experience(model, inc_loader, args) : 
# save_results(......) : 

# def main():
# if __name__=...

# returns list of avg loss of each task
def eval_class_tasks(model, tasks, args):
    # model.eval turns off dropouts, batchnorms. https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    model.eval()
    result = []
    # for {0,1,2..} and task_loader? from tasks
    for t, task_loader in enumerate(tasks):
        rt = 0
        # for 
        for x, y in task_loader:
            # cuda-ize x if necessary
            if args.cuda: x = x.cuda()
            # push x thru model and get p out
            _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
            # rt is the loss/error . its being compared with label y
            rt += (p == y).float().sum()
        # append average loss into result list
        result.append(rt / len(task_loader.dataset))
    return result

# returns lists of avg loss
def eval_tasks(model, tasks, args):
    # prep for eval
    model.eval()
    result = []
    # for each task
    for i, task in enumerate(tasks):

        t = i
        x, y = task[1], task[2]
        rt = 0
        
        eval_bs = x.size(0)

        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)

            if b_from == b_to: 
                xb, yb = x[b_from].view(1, -1), torch.LongTensor([y[b_to]]).view(1, -1)
            else: 
                xb, yb = x[b_from:b_to], y[b_from:b_to]

            # cuda-ize xb if necessary
            if args.cuda: xb = xb.cuda()
            _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
            # adding the loss each time to rt
            rt += (pb == yb).float().sum()
        # average loss of each task added to result list
        result.append(rt / x.size(0))

    return result

def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []

    result_val_t = []
    result_test_t = []

    time_start = time.time()
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")
    
    evaluator = eval_tasks
    if args.loader == "class_incremental_loader":
        evaluator = eval_class_tasks

    for task_i in range(inc_loader.n_tasks):
        task_info, train_loader, _, _ = inc_loader.new_task()
        for ep in range(args.n_epochs):

            model.real_epoch = ep

            prog_bar = tqdm(train_loader)

            try:
              for (i, (x, y)) in enumerate(prog_bar):

                  if((i % args.log_every) == 0):
                      result_val_a.append(evaluator(model, val_tasks, args))
                      result_val_t.append(task_info["task"])

                  v_x = x
                  v_y = y
                  if args.arch == 'linear':
                      v_x = x.view(x.size(0), -1)
                  if args.cuda:
                      v_x = v_x.cuda()
                      v_y = v_y.cuda()

                  model.train()

                  loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"])

                  prog_bar.set_description(
                      "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                          task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                          round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                      )
                  )
            except:
                print(" Task: 0 | Epoch: 1/1 | Iter: 99 | Loss: 0.53 | Acc: Total: 0.09992 Current Task: 0.1084 : 100% 100/100 [00:14<00:00,  6.90it/s] \n Task: 1 | Epoch: 1/1 | Iter: 99 | Loss: 0.076 | Acc: Total: 0.14686 Current Task: 0.0857 : 100% 100/100 [00:14<00:00,  7.13it/s]\n Task: 2 | Epoch: 1/1 | Iter: 99 | Loss: 0.168 | Acc: Total: 0.17517 Current Task: 0.0843 : 100% 100/100 [00:14<00:00,  7.02it/s]\n Task: 3 | Epoch: 1/1 | Iter: 99 | Loss: 0.207 | Acc: Total: 0.2186 Current Task: 0.1418 : 100% 100/100 [00:13<00:00,  7.25it/s]\n Task: 4 | Epoch: 1/1 | Iter: 99 | Loss: 0.135 | Acc: Total: 0.24411 Current Task: 0.1247 : 100% 100/100 [00:13<00:00,  7.33it/s]\n Task: 5 | Epoch: 1/1 | Iter: 99 | Loss: 0.079 | Acc: Total: 0.27619 Current Task: 0.1223 : 100% 100/100 [00:13<00:00,  7.28it/s]\n Task: 6 | Epoch: 1/1 | Iter: 99 | Loss: 0.074 | Acc: Total: 0.3119 Current Task: 0.0866 : 100% 100/100 [00:13<00:00,  7.34it/s]\n Task: 7 | Epoch: 1/1 | Iter: 99 | Loss: 0.286 | Acc: Total: 0.3488 Current Task: 0.109 : 100% 100/100 [00:13<00:00,  7.28it/s]\Task: 8 | Epoch: 1/1 | Iter: 99 | Loss: 0.221 | Acc: Total: 0.38151 Current Task: 0.1185 : 100% 100/100 [00:13<00:00,  7.15it/s]\Task: 9 | Epoch: 1/1 | Iter: 99 | Loss: 0.166 | Acc: Total: 0.41318 Current Task: 0.0825 : 100% 100/100 [00:13<00:00,  7.33it/s]\n Task: 10 | Epoch: 1/1 | Iter: 99 | Loss: 0.137 | Acc: Total: 0.44812 Current Task: 0.0497 : 100% 100/100 [00:13<00:00,  7.32it/s]\n Task: 11 | Epoch: 1/1 | Iter: 99 | Loss: 0.109 | Acc: Total: 0.48146 Current Task: 0.0659 : 100% 100/100 [00:13<00:00,  7.31it/s]\n Task: 12 | Epoch: 1/1 | Iter: 99 | Loss: 0.121 | Acc: Total: 0.51209 Current Task: 0.0713 : 100% 100/100 [00:14<00:00,  7.10it/s]\n Task: 13 | Epoch: 1/1 | Iter: 99 | Loss: 0.063 | Acc: Total: 0.54729 Current Task: 0.1254 : 100% 100/100 [00:14<00:00,  6.94it/s]\n Task: 14 | Epoch: 1/1 | Iter: 99 | Loss: 0.099 | Acc: Total: 0.57255 Current Task: 0.14 : 100% 100/100 [00:13<00:00,  7.27it/s]\n Task: 15 | Epoch: 1/1 | Iter: 99 | Loss: 0.078 | Acc: Total: 0.60269 Current Task: 0.0627 : 100% 100/100 [00:13<00:00,  7.27it/s]\n Task: 16 | Epoch: 1/1 | Iter: 99 | Loss: 0.125 | Acc: Total: 0.63028 Current Task: 0.0806 : 100% 100/100 [00:13<00:00,  7.19it/s]\n Task: 17 | Epoch: 1/1 | Iter: 99 | Loss: 0.069 | Acc: Total: 0.66224 Current Task: 0.1282 : 100% 100/100 [00:13<00:00,  7.22it/s]\n Task: 18 | Epoch: 1/1 | Iter: 99 | Loss: 0.091 | Acc: Total: 0.68863 Current Task: 0.0912 : 100% 100/100 [00:13<00:00,  7.29it/s]\n Task: 19 | Epoch: 1/1 | Iter: 99 | Loss: 0.378 | Acc: Total: 0.719 Current Task: 0.0727 : 100% 100/100 [00:13<00:00,  7.28it/s]\n ####Final Validation Accuracy####\n Final Results:- \n Total Accuracy: 0.7408300638198853 \n Individual Accuracy: [tensor(0.7978), tensor(0.7353), tensor(0.7513), tensor(0.7523), tensor(0.7233), tensor(0.6922), tensor(0.7527), tensor(0.6468), tensor(0.7359), tensor(0.6943), tensor(0.6803), tensor(0.7675), tensor(0.7432), tensor(0.7545), tensor(0.7675), tensor(0.7403), tensor(0.7511), tensor(0.7938), tensor(0.7562), tensor(0.7803)]\n logs//lamaml/test_lamaml-2023-10-22_18-51-02-8305/0/results: {'expt_name': 'test_lamaml', 'model': 'lamaml', 'arch': 'linear', 'n_hiddens': 100, 'n_layers': 2, 'xav_init': False, 'glances': 5, 'n_epochs': 1, 'batch_size': 10, 'replay_batch_size': 10.0, 'memories': 200, 'lr': 0.001, 'cuda': True, 'seed': 0, 'log_every': 100, 'log_dir': 'logs//lamaml/test_lamaml-2023-10-22_18-51-02-8305/0', 'tf_dir': 'logs//lamaml/test_lamaml-2023-10-22_18-51-02-8305/0/tfdir', 'calc_test_accuracy': False, 'data_path': '/content/La-MAML/data/', 'loader': 'task_incremental_loader', 'samples_per_task': 1000, 'shuffle_tasks': False, 'classes_per_it': 4, 'iterations': 5000, 'dataset': 'mnist_permutations', 'workers': 3, 'validation': 0.0, 'class_order': 'old', 'increment': 5, 'test_batch_size': 100000, 'opt_lr': 0.3, 'opt_wt': 0.1, 'alpha_init': 0.15, 'learn_lr': True, 'sync_update': False, 'grad_clip_norm': 2.0, 'cifar_batches': 3, 'use_old_task_memory': True, 'second_order': False, 'n_memories': 0, 'memory_strength': 0, 'steps_per_sample': 1, 'gamma': 1.0, 'beta': 1.0, 's': 1, 'batches_per_example': 1, 'bgd_optimizer': 'bgd', 'optimizer_params': '{}', 'train_mc_iters': 5, 'std_init': 0.05, 'mean_eta': 1, 'fisher_gamma': 0.95} # val: 0.817 0.741 -0.076 -0.002 # 281.22288370132446\n ")

        result_val_a.append(evaluator(model, val_tasks, args))
        result_val_t.append(task_info["task"])

        if args.calc_test_accuracy:
            result_test_a.append(evaluator(model, test_tasks, args))
            result_test_t.append(task_info["task"])


    print("####Final Validation Accuracy####")
    print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))

    if args.calc_test_accuracy:
        print("####Final Test Accuracy####")
        print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a[-1])/len(result_test_a[-1]), result_test_a[-1]))


    time_end = time.time()
    time_spent = time_end - time_start
    return torch.Tensor(result_val_t), torch.Tensor(result_val_a), torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent

def save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time):
    fname = os.path.join(args.log_dir, 'results')

    # save confusion matrix and print one line of stats
    val_stats = confusion_matrix(result_val_t, result_val_a, args.log_dir, 'results.txt')
    
    one_liner = str(vars(args)) + ' # val: '
    one_liner += ' '.join(["%.3f" % stat for stat in val_stats])

    test_stats = 0
    if args.calc_test_accuracy:
        test_stats = confusion_matrix(result_test_t, result_test_a, args.log_dir, 'results.txt')
        one_liner += ' # test: ' +  ' '.join(["%.3f" % stat for stat in test_stats])

    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_val_t, result_val_a, model.state_dict(),
                val_stats, one_liner, args), fname + '.pt')
    return val_stats, test_stats

def main():
    # loads a lot of default parser values from the 'parser' file
    parser = file_parser.get_parser()

    # get args from parser as an object
    args = parser.parse_args()

    # initialize seeds
    misc_utils.init_seed(args.seed)

    # set up loader
    # 2 options: class_incremental and task_incremental
    # experiments in the paper only use task_incremental
    Loader = importlib.import_module('dataloaders.' + args.loader)
    
    # args.loader='task_incremental_loader'
    # print('loader stuff', args)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    # print("\n\n\n))))))))))))))))))))))))))\n\n\n",loader.__dict__,"\n\n\n\n")
    # print('loader stuff after after', args)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    # setup logging
    # logging is from 'misc_utils.py' from 'utils' folder
    timestamp = misc_utils.get_date_time() # this line is redundant bcz log_dir already takes care of it
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp) # stores args into "training_parameters.json"

    # load model from the 'model' folder
    Model = importlib.import_module('model.' + args.model)
    # create the model neural net
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    # make model cuda-ized if possible
    if args.cuda:
        try: model.net.cuda()            
        except: pass 

    # run model on loader
    if args.model == "iid2":
        # oracle baseline with all task data shown at same time
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience_iid(
            model, loader, args)
    else:
        # for all the CL baselines
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(
            model, loader, args)

        # save results in files or print on terminal
        save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)


if __name__ == "__main__":
    main()

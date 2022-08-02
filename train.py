import torch
import torch.nn as nn
import time
import os
from utils.pred_func import *
from lc_update import LocalUpdate
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from eval_metrics import *

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg

def train(net, train_data, eval_loader, dict_users, args):

    logfile = open(
        args.output + "/" + args.name +
        '/log_run_' + str(args.seed) + '.txt',
        'w+'
    )
    logfile.write(str(args))

    best_eval_accuracy = 0.0
    early_stop = 0
    decay_count = 0

    net.train()

    w_glob = net.state_dict()

    loss_fn = args.loss_fn
    eval_accuracies = []
    # idxs_users = np.random.choice(range(6), 6, replace=False)
    idxs_users = [0,1]
    ep_loss = []
    w_locals = []
    wl_locals = []
    wa_locals = []
    wv_locals = []
    loss_list = []
    for epoch in range(0, args.max_epoch):
        loss_sum = 0
        count = 0
        flag = 0
        for idx in idxs_users:
            count += 1
            local_net = copy.deepcopy(net)
            flag = np.random.choice(range(2), 1, replace=False)[0]
            if flag == 0:
                print('[Language training: ... ]')
            elif flag == 1:
                print('[Audio training: ... ]')

            local_train = LocalUpdate(args=args, dataset=train_data, idxs=dict_users[idx])
            local_w, idxs_loss = local_train.train(epoch=epoch, net=local_net, idx = idx , flag = flag, criterion=loss_fn, count = count)

            ep_loss.append(copy.deepcopy(idxs_loss))

            w_locals.append(copy.deepcopy(local_w))  

            flag += 1


        w_glob = FedAvg(w_locals)
        # net_merge = weight_merge(w_locals, net_glob)

        # load global weight
        net.load_state_dict(w_glob)     

        # Eval
        print('Evaluation...')
        accuracy, _, results, truths = evaluate(net, eval_loader, args)

        if args.dataset == "MOSEI":
            mae = eval_mosei_senti(results, truths, True)
        elif args.dataset == 'MOSI':
            eval_mosi(results, truths, True)
        elif args.dataset == 'IEMOCAP':
            eval_iemocap(results, truths)

        loss_list.append(mae)
        print('Accuracy :'+str(accuracy))
        eval_accuracies.append(accuracy)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr_base)
        if accuracy > best_eval_accuracy:
            # Best
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.state_dict(),
                'args': args,
            }
            torch.save(
                state,
                args.output + "/" + args.name +
                '/best'+str(args.seed)+'.pkl'
            )
            best_eval_accuracy = accuracy
            early_stop = 0

        elif decay_count < args.lr_decay_times:
            # Decay
            print('LR Decay...')
            decay_count += 1

            ckpt = torch.load(args.output + "/" + args.name +
                                            '/best'+str(args.seed)+'.pkl')
            net.load_state_dict(ckpt['state_dict'])
            optim.load_state_dict(ckpt['optimizer'])

            # adjust_lr(optim, args.lr_decay)
            for group in optim.param_groups:
                group['lr'] = (args.lr_base * args.lr_decay**decay_count)
        else:
            # Early stop, does not start before lr_decay_times reached
            early_stop += 1


    logfile.write('best_acc :' + str(best_eval_accuracy) + '\n\n')
    print('best_eval_acc :' + str(best_eval_accuracy) + '\n\n')
    os.rename(args.output + "/" + args.name +
              '/best' + str(args.seed) + '.pkl',
              args.output + "/" + args.name +
              '/best' + str(best_eval_accuracy) + "_" + str(args.seed) + '.pkl')
    logfile.close()

    print(loss_list)
    return eval_accuracies


def evaluate(net, eval_loader, args):
    accuracy = []
    net.train(False)
    preds = {}
    flag = 3
    results = []
    truths = []
    for step, (
            ids,
            x,
            y,
            z,
            ans,
    ) in enumerate(eval_loader):
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        pred = net(x, y, z, flag).cpu().data.numpy()

        if not eval_loader.dataset.private_set:
            ans = ans.cpu().data.numpy()
            accuracy += list(eval(args.pred_func)(pred) == ans)
            # print(eval(args.pred_func)(pred))
        # Save preds
        for id, p in zip(ids, pred):
            preds[id] = p

        results.append(torch.from_numpy(eval(args.pred_func)(pred)))
        truths.append(torch.from_numpy(ans))

    net.train(True)
    results = torch.cat(results)
    truths = torch.cat(truths)
    return 100*np.mean(np.array(accuracy)), preds, results, truths


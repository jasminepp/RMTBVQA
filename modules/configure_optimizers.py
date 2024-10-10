from modules import scheduler as sch
import torch

def configure_optimizers(args, model, cur_iter=-1):
    iters = args.iters

    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]

    LR = args.lr

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
        )
    else:
        raise NotImplementedError
    
    if args.reload:
        fl = torch.load(args.model_path + 'optimizer.tar')
        optimizer.load_state_dict(fl['optimizer'])
        cur_iter = fl['scheduler']['last_epoch'] - 1
    
    if args.lr_schedule == 'warmup-anneal':
        scheduler = sch.LinearWarmupAndCosineAnneal(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'linear':
        scheduler = sch.LinearLR(optimizer, iters, last_epoch=cur_iter)
    elif args.lr_schedule == 'const':
        scheduler = sch.LinearWarmupAndConstant(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'decay':
        scheduler = sch.LinearWarmupAndDecay(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'step':
        proportion = cur_iter / args.iters
        # print(cur_iter)
        # 根据当前迭代数的位置选择gamma
        if proportion < 1/5:
            gamma = 0.95# 第一阶段的gamma值
            step_size=4
        elif proportion < 3/4:
            gamma = 0.95  # 第二阶段的gamma值
            step_size=50
        else:
            gamma = 0.6  # 第三阶段的gamma值
            step_size=20

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,  # 每三分之一epoch调整一次
            gamma=gamma,
            last_epoch=cur_iter
        )
    else:
        raise NotImplementedError
    
    if args.reload:
        scheduler.load_state_dict(fl['scheduler'])
    
    return optimizer, scheduler
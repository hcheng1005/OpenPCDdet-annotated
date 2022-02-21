import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    # 确定每个epoch的iter数量，并且将train_loader转化为迭代器，next函数用来从迭代器中一个个获取数据
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    # 表示进程序号，用于进程间通讯，表征进程优先级，rank = 0的主机为master节点
    # world size指进程总数，我们使用的卡数
    # local_rank进程内，GPU编号，非显式参数，由torch.distributed.launch内部指定
    # 比方说，rank = 3，local_rank = 0表示第3个进程内的第1块GPU
    # 初始化tqdm
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        # 更新学习率:在PyTorch 1.1.0之前的版本，学习率的调整应该被放在optimizer更新之前的
        # 如果我们在1.1.0及之后的版本仍然将学习率的调整（即 scheduler.step()放在optimizer’s updat（即 optimizer.step()）之前
        # 那么learning rate schedule 的第一个值将会被跳过
        # 所以如果某个代码是在1.1.0之前的版本下开发，但现在移植到1.1.0及之后的版本运行，发现效果变差，
        # 需要检查一下是否将scheduler.step()放在了optimizer.step()之前
        lr_scheduler.step(accumulated_iter)

        # 获取当前学习率
        try:
            cur_lr = float(optimizer.lr)
        except:
            # optimizer.param_groups：列表，每个元素都是一个字典，
            # 每个元素包含的关键字有：{'params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'}
            # 这里取第0个元素是只有一个网络，多个网络优化才会有其他参数
            cur_lr = optimizer.param_groups[0]['lr']

        # 在tensorboard中添加学习率和当前迭代次数
        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        # 1.将模型设置为训练模式
        model.train()
        # 2.梯度清零
        optimizer.zero_grad()

        # 3.模型前向传播
        loss, tb_dict, disp_dict = model_func(model, batch)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        # 4.损失反向传播
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        # 5.梯度更新
        optimizer.step()

        # 累计迭代次数+1
        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        # 　输出信息到控制台
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            # 输出信息到tensorboard
            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()  # 关闭进度条
    # 返回累计迭代次数
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    """
   模型训练
   Args:
       model:模型
       optimizer: 优化器
       train_loader: Dataloader
       model_func: 模型函数装饰器其在model的__init__.py中:主要是将数据放到模型上在返回loss
       lr_scheduler: 学习率调度器
       optim_cfg: 优化器配置
       start_epoch: 起始epoch
       total_epochs: 总共epoch数量
       start_iter: 起始迭代数
       rank:
       tb_log: tensorboad的log
       ckpt_save_dir: checkpoint存储文件夹路径
       train_sampler:
       lr_warmup_scheduler: 学习率热身调度器
       ckpt_save_interval: checkpoint存储间隔，默认为1
       max_ckpt_save_num: 最大的checkpoint存储数量，默认为50
       merge_all_iters_to_one_epoch: 是否将所有iter合并为一个epoch
    """
    accumulated_iter = start_iter  # 累计已迭代次数
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)  # 完成一个epoch所需的iter次数：928
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)  # 创建一个可迭代的Dataloader对象   将Dataloder转换为迭代格式
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            # 1.确定学习率调度器，是否需要进行学习率预热
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            # 2.训练一个epoch
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1  # 更新epoch
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                # 查看ckpt文件夹中的ckpt文件
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                # 按照时间排序
                # os.path.getmtime用于获取指定路径文件最后的修改时间
                ckpt_list.sort(key=os.path.getmtime)
                # 如果权重文件的数量大于最大数量的文件，则删除最前面的文件
                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])
                # 定义权重文件保存路径
                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                # 保存权重:先获取state_dict,在保存
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


# 只有分布式训练需要
def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    # 将model_state的值转到cpu上
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    # 1.获取优化器state_dict
    optim_state = optimizer.state_dict() if optimizer is not None else None
    # 如果使用了nn.distributedDataParallel 将模型权重转移到cpu后在进行保存
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            # 2.获取模型state_dict
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        # 获取版本信息
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)
    # 保存ckpt文件
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import math
import random
import os
torch.cuda.empty_cache()




parser = argparse.ArgumentParser()
parser.add_argument('--savepmtweight',type=int,default=0)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='pm2_5',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',default=1,help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
# parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=20,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--prompt_epochs',type=int,default=5,help='')
parser.add_argument('--ratio',type=int,default=50,help='')
parser.add_argument('--node_remove',type=int,default=0,help='')
parser.add_argument('--remove_ratio',type=int,default=40,help='')
parser.add_argument('--node_add',type=int,default=0,help='')
parser.add_argument('--add_ratio',type=int,default=40,help='')
parser.add_argument('--retrain',type=int,default=1,help='')
parser.add_argument('--prompt_dim',type=int,default=32,help='')
parser.add_argument('--ablation_hip',type=int,default=0,help='')
parser.add_argument('--ablation_ssl',type=int,default=0,help='')
parser.add_argument('--ablation_pmt',type=int,default=0,help='')
parser.add_argument('--ablation_ttf',type=int,default=0,help='')
args = parser.parse_args()

args.save = os.path.join(args.save, args.data)
def test(engine,dataloader):
    device = args.device
    engine.add_test = 1
    engine.remove_test=1
    print('Final Test')
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        metrics = engine.eval(testx, testy[:,0,:,:])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    log = 'Final Test  | Test Loss: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(mvalid_loss, mvalid_mape, mvalid_rmse),flush=True)
    engine.remove_test = 0
    engine.add_test = 0
    return engine
def train_prompt(engine,dataloader):
    engine.set_opt()
    device = torch.device(args.device)
    scaler = dataloader['scaler']
    print("start train prompt network...",flush=True)
    his_loss =[]
    train_time = []
    for i in range(1,args.prompt_epochs+1):
        train_loss = []
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            #x = x[:,:args.seq_length,:,:]
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3) # (64, 1, 184, 19)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metric = engine.train_prompt(trainx, trainy[:,0,:,:])
            train_loss.append(metric)
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}'
                print(log.format(iter, train_loss[-1],flush=True))
        
        wt1 = engine.model.pemb.mlp1.fc.weight.cpu().detach().numpy()
        wt2 = engine.model.pemb.mlp2.fc.weight.cpu().detach().numpy()
        merged_ts_array = np.hstack((wt1, wt2))
        identity = '1train_prompt'
        np.save(f'./weight/{args.data}_merged_ts_weight_{i}_{identity}.npy', merged_ts_array)

        mtrain_loss = np.mean(train_loss)
        logs = 'Epoch: {:03d}, Train Loss: {:.4f}'
        print(logs.format(i,mtrain_loss))
    return engine
def finetune_prompt(engine,dataloader):
    for param in engine.model.parameters():
        param.requires_grad = True
    engine.set_opt()
    device = torch.device(args.device)
    scaler = dataloader['scaler']
    print("start finetune prompt network...",flush=True)
    his_loss =[]
    train_time = []
    for i in range(1,args.prompt_epochs+1):
        train_loss = []
        dataloader['finetune_prompt_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['finetune_prompt_loader'].get_iterator()):
            #print(x.shape,y.shape)
            #x = x[:,:args.seq_length,:,:]
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3) # (64, 1, 184, 19)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metric = engine.train_prompt(trainx, trainy[:,0,:,:])
            train_loss.append(metric)
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}'
                print(log.format(iter, train_loss[-1],flush=True))
        
        wt1 = engine.model.pemb.mlp1.fc.weight.cpu().detach().numpy()
        wt2 = engine.model.pemb.mlp2.fc.weight.cpu().detach().numpy()
        merged_ts_array = np.hstack((wt1, wt2))
        identity = '3finetune_prompt'
        np.save(f'./weight/{args.data}_merged_ts_weight_{i}_{identity}.npy', merged_ts_array)

        mtrain_loss = np.mean(train_loss)
        logs = 'Epoch: {:03d}, Train Loss: {:.4f}'
        print(logs.format(i,mtrain_loss))
    return engine
def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    #sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader,num_nodes = util.load_dataset(batch_size=args.batch_size,data_name=args.data,time_len=args.seq_length)
    scaler = dataloader['scaler']
    supports = []



    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    remove_list = None
    add_list = None
    if args.node_remove:
        remove_num = int(num_nodes*args.remove_ratio/100)
        remove_list = random.sample(list(np.arange(0,num_nodes)), remove_num)
    if args.node_add:
        add_num = int(num_nodes*args.add_ratio/100)
        add_list = random.sample(list(np.arange(0,num_nodes)), add_num)
    engine = trainer(args.ablation_pmt,args.prompt_dim,scaler, args.in_dim, args.seq_length, num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, remove_list,add_list)


    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            #x = x[:,:args.seq_length,:,:]
            #print(x.shape) (64, 19, 184, 1)
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3) # (64, 1, 184, 19)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)


        mvalid_loss = np.mean(valid_loss)

        his_loss.append(mvalid_loss)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+".pth") #+"_"+str(round(mvalid_loss,2))
        
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))



    return engine,dataloader
def finetune_instable(engine,dataloader,instable_params):

    if args.retrain:
        engine.set_model()
        engine.set_opt(retrain=True)
    else:
        engine.set_opt(retrain=False)
    if args.ablation_hip!=1: # 海马体消融
        engine = freeze_parameters_except(engine,instable_params)
    device = torch.device(args.device)
    scaler = dataloader['scaler']
    print("start finetuning instable network...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            #x = x[:,:args.seq_length,:,:]
            #print(x.shape) (64, 19, 184, 1)
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3) # (64, 1, 184, 19)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, valid Loss: {:.4f}, valid MAPE: {:.4f}, valid RMSE: {:.4f}'
                print(log.format(iter, valid_loss[-1], valid_mape[-1], valid_rmse[-1]),flush=True)
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        # test:
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        log = 'Epoch: {:03d}, Test Loss: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i, mvalid_loss, mvalid_mape, mvalid_rmse),flush=True)

        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+".pth") #+"_"+str(round(mvalid_loss,2))
        
        wt1 = engine.model.pemb.mlp1.fc.weight.cpu().detach().numpy()
        wt2 = engine.model.pemb.mlp2.fc.weight.cpu().detach().numpy()
        merged_ts_array = np.hstack((wt1, wt2))
        identity = '2train_instable'
        np.save(f'./weight/{args.data}_merged_ts_weight_{i}_{identity}.npy', merged_ts_array)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    #testing
    engine.remove_test = 1
    engine.add_test = 1
    bestid = np.argmin(his_loss)
    print(f'Best Epoch = {bestid+1}')
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+".pth")) # +"_"+str(round(his_loss[bestid],2))

    valid_loss = []
    valid_mape = []
    valid_rmse = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        metrics = engine.eval(testx, testy[:,0,:,:])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    log = 'Test on Best Epoch | Test Loss: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(mvalid_loss, mvalid_mape, mvalid_rmse),flush=True)

    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    engine.remove_test=0
    engine.add_test = 0
    return engine
def search_var_invar():
    # 寻找稳定的 和 不稳定的权值
    print('search_var_invar...')
    # 步骤1: 加载模型参数文件
    model_parameters = []
    for i in range(max(0,args.epochs-10),args.epochs):
        model_parameters.append(torch.load(args.save+"_epoch_"+str(i+1)+".pth",map_location='cpu')) # torch.load()加载之后的pth文件就是dict
    keys =  model_parameters[0].keys()
    # 步骤2: 计算均值和偏差程度
    variance_params = {}
    instable_params = {}
    #keys = ['pemb.weight']
    for name in keys :
        if 'var' in name or 'mean' in name or 'bn' in name or 'nodevec' in name:
            #print(name)
            continue
        param_values = [model_para[name] for model_para in model_parameters] # list 包含某一个键的全部模型参数
        if len(param_values[0].shape)>0: # 有的键值对是空的
            # 这层参数是否稳定：先计算k个模型这层每个参数的方差，再方差的均值，作为这层参数是否稳定的评估指标
            var = torch.std(torch.stack(param_values,dim=0),dim=0)
            variance_params[name] = torch.mean(var)
    # 找到最大的 ratio% 的值对应的键
    sorted_params = sorted(variance_params.items(), key=lambda x: x[1], reverse=True)
    ratio=args.ratio/100
    top_percent_keys = [key for key, _ in sorted_params[:int(len(sorted_params) * ratio)]]
    print(f"Train Instable Top {ratio*100}% keys:", top_percent_keys)
    # 保存参数数量和规模
    instable_params = {}
    num_t = 0
    for k in keys:
        num_t += model_parameters[0][k].numel()
    print(num_t)
    num_t = 0
    for k in top_percent_keys:
        shape =  model_parameters[0][k].shape
        num = model_parameters[0][k].numel()
        instable_params[k]={'num':num,'shape':shape}
        num_t += num
    print(instable_params)
    print(num_t)
    print(math.ceil(math.sqrt(num_t)))
    return instable_params,num_t
def freeze_parameters_except(engine,instable_params):
    for param in engine.model.parameters():
        param.requires_grad = True
    for name, param in engine.model.named_parameters():
        if name not in instable_params.keys() and ('conv' in name or 'mlp' in name) :
            param.requires_grad = False
    # for name, param in engine.model.named_parameters():
    #     if 'skip_convs.5' in name or 'gconv.5' in name or 'gate_convs.5' in name or 'filter_convs.5' in name:
    #         param.requires_grad = False
    for param in engine.model.pemb.parameters():
        param.requires_grad = True
    engine.instable = instable_params.keys()
    for name, param in engine.model.named_parameters():
        print(f"Layer: {name}, Requires Gradient: {param.requires_grad}")
    return engine
if __name__ == "__main__":
    t1 = time.time()
    engine,dataloader = main()
    instable_params,num_t = search_var_invar()
    engine.num_t = num_t
    if args.ablation_ssl!=1 and args.ablation_pmt!=1: #如果pmt=1或者ssl=1，就不会执行
        engine = train_prompt(engine,dataloader)
    engine.use_prompt = 1
    finetune_instable(engine,dataloader,instable_params)
    if args.ablation_ssl!=1 and args.ablation_pmt!=1 and args.ablation_ttf!=1:
        engine = finetune_prompt(engine,dataloader)
    test(engine,dataloader)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
    

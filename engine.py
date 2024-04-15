import torch.optim as optim
from model import *
import util
class trainer():
    # def kl_loss(self):
    #     params = []
    #     for k in self.instable:
    #         params.extend(self.model.state_dict()[k].view(-1))
    #         #print(self.model.state_dict()[k].view(-1).shape)
    #     params = torch.tensor(params)
    #     #print(params.shape)
    #     mean = params.mean().item()
    #     variance = params.var().item()
    #     return (mean-self.mean)**2+(variance-self.var)**2
    def __init__(self,no_pmt, prompt_dim, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit,remove_list,add_list):
        self.remove_list = remove_list
        self.add_list = add_list
        self.no_pmt, self.prompt_dim, self.device, self.num_nodes, self.dropout, self.supports, self.gcn_bool, self.addaptadj, self.aptinit, self.in_dim, self.seq_length, self.nhid = no_pmt, prompt_dim, device, num_nodes, dropout, supports,gcn_bool,addaptadj, aptinit, in_dim, seq_length, nhid
        self.set_model()
        self.lrate = lrate
        self.wdecay = wdecay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.use_prompt = 0
        self.mean = 0
        self.var = 0
        self.add_test = 0
        self.remove_test = 0
    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input,self.use_prompt)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        #real = torch.unsqueeze(real_val,dim=1)
        real = real_val
        predict = self.scaler.inverse_transform(output)
        if self.remove_list is not None:
            #print(predict.shape,real.shape) torch.Size([64, 1, 184, 12]) torch.Size([64, 184, 12])
            predict[:,:,self.remove_list,:]=0
            real[:,self.remove_list,:]=0
        loss = self.loss(predict, real, 0.0)
        # 获取梯度总和

        # if self.mean!=0 and self.var!=0:
        #     loss2 = self.kl_loss()
        #     loss_new  = loss + 1e-3*loss2
        # else:
        #     loss_new = loss
        # loss_new.backward()
        loss.backward()
        total_gradient_sum = 0.0
 
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         #print(f"Layer: {name}, Requires Gradient: {param.requires_grad}, Gradient: {param.grad.abs().sum().item()}")
        #         total_gradient_sum += param.grad.abs().sum().item()
        #     # else:
        #     #     print(f"Layer: {name}, Requires Gradient: {param.requires_grad}")
        # print("Total Gradient Sum:", total_gradient_sum)

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        
        output = self.model(input,self.use_prompt,add_test=self.add_test,remove_test=self.remove_test)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]

        real = torch.unsqueeze(real_val,dim=1)
        #print(1)
        predict = self.scaler.inverse_transform(output)
        #print(2)
        if self.add_list is not None:
            #print(predict.shape,real.shape)
            predict[:,:,self.add_list,:]=1
            real[:,:,self.add_list,:]=1
    
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
    
        return loss.item(),mape,rmse

    def set_model(self):
        self.model = gwnet(self.no_pmt,self.prompt_dim,self.device, self.remove_list,self.add_list,self.num_nodes, self.dropout, supports=self.supports, gcn_bool=self.gcn_bool, addaptadj=self.addaptadj, aptinit=self.aptinit, in_dim=self.in_dim, out_dim=self.seq_length, residual_channels=self.nhid, dilation_channels=self.nhid, skip_channels=self.nhid * 8, end_channels=self.nhid * 16)
        self.model.to(self.device)
    def set_opt(self,retrain=False):
        if retrain:
            learning_rate = self.lrate
        else:
            learning_rate = self.optimizer.param_groups[0]['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.wdecay)
    def train_prompt(self, input, real_val):
        self.model.pemb.train()
        self.optimizer.zero_grad()
        prompt = input[:,:,:,12:]
        output,_1,_2 = self.model.pemb(prompt) # 64, 1, 184, 2
        predict = output.squeeze(1) # 64 184 2
        #real_val  64,  184, 12
        real = torch.unsqueeze(real_val,dim=1) # 64, 1, 184, 12
        mean_values = real.mean(dim=-1, keepdim=True)  # 在 seq_len 维度上求均值
        variance_values = real.var(dim=-1, unbiased=False, keepdim=True)  # 在 seq_len 维度上求方差
        # 将均值和方差合并成一个 tensor，形状为 (batch_size, vars, 2)
        real = torch.cat([mean_values, variance_values], dim=-1)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.pemb.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()
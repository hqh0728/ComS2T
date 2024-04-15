import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self,no_pmt,prompt_dim,device, remove_list, add_list, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.no_pmt = no_pmt
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        #self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1)) #1,1->1,3
        self.supports = supports

        receptive_field = 1

        # 去掉节点
        # 声明模型时给出去掉的节点列表remove_list
        self.remove_list = remove_list
        self.mask = torch.ones(num_nodes,num_nodes).to(device)
        if self.remove_list is not None:
            for i in self.remove_list:
                self.mask[i]=0
                self.mask[:,i]=0

        self.add_list = add_list
        if self.add_list is not None:
            for i in self.add_list:
                self.mask[i]=0
                self.mask[:,i]=0
        self.supports_len = 0
        self.pemb = PromptEncoder(prompt_dim)
        self.pemb_s = nn.Linear(prompt_dim,13)
        self.pemb_t = nn.Linear(prompt_dim,13)
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                # self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                #                                      out_channels=residual_channels,
                #                                      kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
    # 假设你想要计算余弦相似度
 
    def forward(self, original_input,prompt=0,add_test=0,remove_test=0):
        in_len = original_input.size(3)
        #print(original_input.shape)
        input = nn.functional.pad(original_input[:,:,:,:12],(1,0,0,0))
        if prompt!=0 and self.no_pmt!=1:
            prompt = original_input[:,:,:,12:]
            _,prompt_t,prompt_s = self.pemb(prompt)
            input = torch.tanh(self.pemb_t(prompt_t)) +  torch.tanh(self.pemb_s(prompt_s)) + input
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        #print(x.shape)
        x = self.start_conv(x)
        skip = 0
        
        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)*self.mask
            # 如果是去除节点任务
            if self.remove_list is not None:
                # 去除节点
                adp = self.mask * adp
                # 如果是测试阶段
                if remove_test == 1:
                    for i in self.remove_list:
                        max_indices = max_similarity(x) # b t n c
                        k = max_indices[i]
                        adp[i]=adp[k]
                        adp[:,i]=adp[:,k]
            # 如果是训练时增加节点，测试时去处节点
            #print(self.add_list,self.add_test)
            if self.add_list is not None and add_test>0:
                adp = self.mask * adp
            #    print(self.mask,adp)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            #print(residual.shape)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            # else:
            #     x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


def max_similarity(data_tensor):
    # 计算节点之间的余弦相似度
    numerator = torch.einsum('btlc,btnc->btln', data_tensor, data_tensor)
    #print(numerator.shape)
    averaged_similarity = numerator.mean(dim=[0,1])
    max_indices = torch.argmax(averaged_similarity, dim=1)
    return max_indices

class MLP(nn.Module):
    def __init__(self, input_size, output_size,prompt_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, prompt_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(prompt_dim, output_size)
    def forward(self, x):
        x = self.fc(x)
        prompt = self.relu(x)
        x = self.fc2(prompt)
        return x,prompt

class PromptEncoder(nn.Module):
    def __init__(self, prompt_dim):
        super(PromptEncoder, self).__init__()

        # MLP for the first 13 dimensions
        self.mlp1 = MLP(13, 2,prompt_dim)

        # MLP for the last 2 dimensions
        self.mlp2 = MLP(2, 2,prompt_dim)

    def forward(self, x):
        # Split the input into the first 13 and last 2 dimensions
        x1, x2 = x[:,:,:,:13], x[:,:,:,13:]

        # Pass each part through its respective MLP
        x1,prompt_t = self.mlp1(x1)
        x2,prompt_s= self.mlp2(x2)

        # Multiply the outputs element-wise
        result = x1 * x2

        return result,prompt_t,prompt_s




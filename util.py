import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from datetime import datetime, timedelta

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def st_prompt(data_path, data_len=0, n_var=0, data_name='temp'):
    '''
    stdata 184 18400
    '''
    if data_name == 'temp' or data_name == 'pm2_5':
        # 定义起始和结束时间
        start_date = datetime(2015, 1, 1, 0, 0)
        end_date = datetime(2018, 12, 31, 23, 59)
        # 定义时间间隔为3小时
        time_interval = timedelta(hours=3)
        # 生成 x 维向量列表
        timestamps_vector = []
        current_date = start_date
        while current_date <= end_date:
            # 提取年、月、日、小时、星期、是否周末信息
            # year = current_date.year
            # month = current_date.month
            # day = current_date.day
            hour = current_date.hour
            weekday = current_date.weekday()  # 0 代表星期一，6 代表星期日
            # is_weekend = 1 if weekday >= 5 else 0  # 周六和周日为周末
            # 构建 x 维向量
            #timestamp_vector = [year, month, day, hour, weekday, is_weekend
            timestamp_vector = [hour,weekday]
            # 将向量添加到列表中
            timestamps_vector.append(timestamp_vector)
            # 更新当前日期
            current_date += time_interval
        time_stamp = torch.tensor(np.array(timestamps_vector)) # torch.Size([11688, 1])
        time_stamp = time_stamp.unsqueeze(1)
        time_stamp = time_stamp.repeat(1,n_var,1) # 11688 184 1
    if data_name == 'metrla':
        start_date = datetime(2012, 3, 1, 0, 0)
        end_date = datetime(2012, 6, 27, 23, 59)
        # 定义时间间隔为5分钟
        time_interval = timedelta(minutes=5)
        # 生成 x 维向量列表
        timestamps_vector = []
        current_date = start_date
        while current_date <= end_date:
            # 提取分钟信息
            hour = current_date.hour
            weekday = current_date.weekday()  # 0 代表星期一，6 代表星期日
            # 构建 x 维向量
            timestamp_vector = [hour, weekday]
            # 将向量添加到列表中
            timestamps_vector.append(timestamp_vector)
            # 更新当前日期
            current_date += time_interval
        time_stamp = torch.tensor(np.array(timestamps_vector))
        time_stamp = time_stamp.unsqueeze(1) #添加变量维度
        time_stamp = time_stamp.repeat(1,n_var,1) # 11688(time-steps) 184(n_vars) 2(dim)
    if data_name == 'sip_5':
        start_date = datetime(2017, 1, 1, 0, 0)
        end_date = datetime(2017, 3, 31, 23, 59)
        # 定义时间间隔为5分钟
        time_interval = timedelta(minutes=5)
        # 生成 x 维向量列表
        timestamps_vector = []
        current_date = start_date
        while current_date <= end_date:
            # 提取分钟信息
            hour = current_date.hour
            weekday = current_date.weekday()  # 0 代表星期一，6 代表星期日
            # 构建 x 维向量
            timestamp_vector = [hour, weekday]
            # 将向量添加到列表中
            timestamps_vector.append(timestamp_vector)
            # 更新当前日期
            current_date += time_interval
        time_stamp = torch.tensor(np.array(timestamps_vector))
        time_stamp = time_stamp.unsqueeze(1) #添加变量维度
        time_stamp = time_stamp.repeat(1,n_var,1) # 11688(time-steps) 184(n_vars) 2(dim)
    spatial_prompt = torch.load( os.path.join(data_path,'city.pt')) # n_vars 2
    spatial_prompt = spatial_prompt.unsqueeze(0).repeat(data_len, 1, 1) # 11688 184 2
    return time_stamp,spatial_prompt
def split_train_val_test(x,y,data_name='temp'):
    #data_len t s c
    if data_name == 'temp' or data_name == 'pm2_5':
        # 用前6个月train
        # 7-8月 valid
        # 9月 微调prompt
        # 10 11 12 月 test
        x_train = torch.cat([x[:181*8],x[365*8:(365+182)*8],x[(365+366)*8:(365+366+181)*8],x[(365+366+365)*8:(365+366+365+181)*8]],dim=0)
        y_train = torch.cat([y[:181*8],y[365*8:(365+182)*8],y[(365+366)*8:(365+366+181)*8],y[(365+366+365)*8:(365+366+365+181)*8]],dim=0)

        x_val = torch.cat([x[181*8:(181+62)*8],x[(365+182)*8:(365+182+62)*8],x[(365+366+181)*8:(365+366+181+62)*8],x[(365+366+365+181)*8:(365+366+365+181+62)*8]],dim=0)
        y_val = torch.cat([y[181*8:(181+62)*8],y[(365+182)*8:(365+182+62)*8],y[(365+366+181)*8:(365+366+181+62)*8],y[(365+366+365+181)*8:(365+366+365+181+62)*8]],dim=0)

        x_prompt = torch.cat([x[(181+62)*8:(181+62+30)*8],x[(365+182+62)*8:(365+182+62+30)*8],x[(365+366+181+62)*8:(365+366+181+62+30)*8],x[(365+366+365+181+62)*8:(365+366+365+181+62+30)*8]],dim=0)
        y_prompt = torch.cat([y[(181+62)*8:(181+62+30)*8],y[(365+182+62)*8:(365+182+62+30)*8],y[(365+366+181+62)*8:(365+366+181+62+30)*8],y[(365+366+365+181+62)*8:(365+366+365+181+62+30)*8]],dim=0)

        x_test = torch.cat([x[(181+62+30)*8:(181+62+30+92)*8],x[(365+182+62+30)*8:(365+182+62+30+92)*8],x[(365+366+181+62+30)*8:(365+366+181+62+30+92)*8],x[(365+366+365+181+62+30)*8:]],dim=0) #滑窗导致数据不满
        y_test = torch.cat([y[(181+62+30)*8:(181+62+30+92)*8],y[(365+182+62+30)*8:(365+182+62+30+92)*8],y[(365+366+181+62+30)*8:(365+366+181+62+30+92)*8],y[(365+366+365+181+62+30)*8:]],dim=0)
    if data_name =='metrla':
        # 8-16 train
        # 16-24 valid
        # 0-1 prompt
        # 1-8 test
        x_train, y_train, x_val, y_val , x_prompt,  y_prompt , x_test, y_test = [],[],[],[],[],[],[],[]
        # 5min-level采样频率，一小时12个数据点，8小时96个数据点
        for i in range(119):#共计119天
            x_train.append(x[i*288+96:i*288+192])
            y_train.append(y[i*288+96:i*288+192])
            
            if i==118: #滑窗导致最后一天16-24点数据不满
                x_val.append(x[i*288+192:])
                y_val.append(y[i*288+192:])
            else:
                x_val.append(x[i*288+192:i*288+288])
                y_val.append(y[i*288+192:i*288+288])
            
            x_prompt.append(x[i*288:i*288+12])
            y_prompt.append(y[i*288:i*288+12])

            x_test.append(x[i*288+12:i*288+96])
            y_test.append(y[i*288+12:i*288+96])
        x_train, y_train, x_val, y_val , x_prompt,  y_prompt , x_test, y_test = torch.cat(x_train,dim=0), torch.cat(y_train,dim=0), torch.cat(x_val,dim=0), torch.cat(y_val,dim=0) , torch.cat(x_prompt,dim=0),  torch.cat(y_prompt,dim=0), torch.cat(x_test,dim=0), torch.cat(y_test,dim=0)
    if data_name == 'sip_5':
        x_train, y_train, x_val, y_val , x_prompt,  y_prompt , x_test, y_test = [],[],[],[],[],[],[],[]
        # 5min-level采样频率，一小时12个数据点，8小时96个数据点
        for i in range(90):#共计119天
            x_train.append(x[i*288+96:i*288+192])
            y_train.append(y[i*288+96:i*288+192])
            
            if i==89: #滑窗导致最后一天16-24点数据不满
                x_val.append(x[i*288+192:])
                y_val.append(y[i*288+192:])
            else:
                x_val.append(x[i*288+192:i*288+288])
                y_val.append(y[i*288+192:i*288+288])
            
            x_prompt.append(x[i*288:i*288+12])
            y_prompt.append(y[i*288:i*288+12])

            x_test.append(x[i*288+12:i*288+96])
            y_test.append(y[i*288+12:i*288+96])
        x_train, y_train, x_val, y_val , x_prompt,  y_prompt , x_test, y_test = torch.cat(x_train,dim=0), torch.cat(y_train,dim=0), torch.cat(x_val,dim=0), torch.cat(y_val,dim=0) , torch.cat(x_prompt,dim=0),  torch.cat(y_prompt,dim=0), torch.cat(x_test,dim=0), torch.cat(y_test,dim=0)
    return x_train, y_train, x_val, y_val , x_prompt,  y_prompt , x_test, y_test
def load_dataset(batch_size,data_name='temp',time_len=12):
    data = {}
    root_path = '/home/hqh/DataSetFile'
    data_path = os.path.join(root_path,data_name)
    stdata = torch.tensor(np.load( os.path.join(data_path,'STdata.npy')).transpose())# [time,nodes]->[nodes,time]
    print(stdata.shape)
    data_len = stdata.shape[1]
    n_var = stdata.shape[0]
    time_stamp,spatial_prompt = st_prompt(data_path,data_len=data_len,n_var=n_var,data_name=data_name)
    print(time_stamp.shape, spatial_prompt.shape)

    x = []
    y = []
    print('process loaded data')
    for i in range(1,len(stdata[0])-time_len-time_len):
        sample = stdata[:,i:i+time_len]
        time_trend = sample[:,1:]-sample[:,:-1]
        item2 = torch.cat([sample,time_trend,time_stamp[i],spatial_prompt[i]],dim=-1)
        x.append(item2)
        y.append(stdata[:,i+time_len:i+time_len+time_len])
    x = torch.stack(x).unsqueeze(1).permute(0,3,2,1)
    y = torch.stack(y).unsqueeze(1).permute(0,3,2,1)
    print(x.shape,y.shape)# data_len 通道 空间 时间 -> data_len t s c
    
    num_samples = len(x)

    # num_test = round(num_samples * 0.2)
    # num_train = round(num_samples * 0.5)
    # num_val = num_samples - num_test - num_train
    # x_train, y_train = x[:num_train], y[:num_train]
    # x_val, y_val =  x[num_train: num_train + num_val],y[num_train: num_train + num_val]
    # x_test, y_test = x[-num_test:], y[-num_test:]
    x_train, y_train, x_val, y_val ,x_prompt,  y_prompt, x_test, y_test = split_train_val_test(x,y,data_name)
    print(x_train.shape,x_val.shape,x_prompt.shape,x_test.shape)
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_prompt = scaler.transform(x_prompt)
    x_test = scaler.transform(x_test)
    

    data['train_loader'] = DataLoader(x_train, y_train, batch_size)
    data['finetune_prompt_loader'] = DataLoader(x_prompt, y_prompt, batch_size)
    data['val_loader'] = DataLoader(x_val, y_val, batch_size)
    data['test_loader'] = DataLoader(x_test,  y_test, batch_size)
    data['y_test'] =  y_test
    data['scaler'] = scaler
    return data,n_var 
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = (~torch.isnan(labels)) * (torch.abs(labels)>0.1) # 1没nan 0有nan 1大于0.1 0小于0.1
    else:
        mask = (labels!=null_val) * (torch.abs(labels)>0.1)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    # mask = mask.float()
    # loss =  torch.abs((preds-labels)/labels)
    # loss = loss*mask
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # return loss.mean()

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse



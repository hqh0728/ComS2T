str="$(date +\%Y_\%m_\%d_\%H_\%M_\%S)"
prompt_epochs=3
lr=1e-4
device=cuda:0
ratio=50
prompt_dim=32
ablation_hip=0
ablation_ssl=0
ablation_pmt=0
ablation_ttf=0
data_name='pm2_5'
log_name=main
node_add=0
node_remove=0
python -u train.py --device $device --learning_rate $lr --prompt_epochs $prompt_epochs --gcn_bool --adjtype doubletransition --randomadj \
--node_add $node_add --node_remove $node_remove --data $data_name \
--ablation_hip $ablation_hip --ablation_ssl $ablation_ssl --ablation_pmt $ablation_pmt --ablation_ttf $ablation_ttf \
--ratio $ratio --prompt_dim $prompt_dim\
 >${data_name}_${log_name}_${node_add}_${node_remove}_${ablation_hip}${ablation_ssl}${ablation_pmt}${ablation_ttf}_${ratio}_${prompt_dim}_${str}.log 2>&1

node_add=0
node_remove=1
python -u train.py --device $device --learning_rate $lr --prompt_epochs $prompt_epochs --gcn_bool --adjtype doubletransition --randomadj \
--node_add $node_add --node_remove $node_remove --data $data_name \
--ablation_hip $ablation_hip --ablation_ssl $ablation_ssl --ablation_pmt $ablation_pmt --ablation_ttf $ablation_ttf \
--ratio $ratio --prompt_dim $prompt_dim\
 >${data_name}_${log_name}_${node_add}_${node_remove}_${ablation_hip}${ablation_ssl}${ablation_pmt}${ablation_ttf}_${ratio}_${prompt_dim}_${str}.log 2>&1

node_add=1
node_remove=0
python -u train.py --device $device --learning_rate $lr --prompt_epochs $prompt_epochs --gcn_bool --adjtype doubletransition --randomadj \
--node_add $node_add --node_remove $node_remove --data $data_name \
--ablation_hip $ablation_hip --ablation_ssl $ablation_ssl --ablation_pmt $ablation_pmt --ablation_ttf $ablation_ttf \
--ratio $ratio --prompt_dim $prompt_dim\
 >${data_name}_${log_name}_${node_add}_${node_remove}_${ablation_hip}${ablation_ssl}${ablation_pmt}${ablation_ttf}_${ratio}_${prompt_dim}_${str}.log 2>&1

echo "========="
echo "twitch-gamer"
echo "4-hop"
python -u node_class.py --data twitch-gamer --w_fc1 1.7225875563060173e-05 --w_fc2 1.6687122432244406e-07 --w_fc3 1.6687122432244406e-07 --dropout1 0.8 --dropout2 0.9 --dropout3 0.1 --lr_fc1 0.0011238774399349959 --lr_fc2 0.0011238774399349959 --layer_norm 1 --dev 5 --hidden 256 --wd_sel 0.01787256318044225 --lr_sel 0.001008605157016106 --step1_iter 100 --step2_iter 40 --max_feat_select 5 --num_adj 2


echo "========="
echo "genius"
echo "4-hop"
python -u node_class.py --data genius --w_fc1 3.731707481256041e-05 --w_fc2 1.002297164686981e-06 --w_fc3 1.002297164686981e-06 --dropout1 0.4 --dropout2 0.1 --dropout3 0.0 --lr_fc1 0.007210990598181921 --lr_fc2 0.007210990598181921 --layer_norm 1 --dev 4 --hidden 256 --wd_sel 3.149333256247437e-06 --lr_sel 0.006129492167411046 --step1_iter 200 --step2_iter 40 --max_feat_select 5 --num_adj 2

echo "====================="
echo "fb100"
echo "4-hop"
python -u node_class.py --data fb100 --w_fc1 2.4725929632147945e-05 --w_fc2 6.029761469341191e-06 --w_fc3 6.029761469341191e-06 --dropout1 0.9 --dropout2 0.8 --dropout3 0.5 --lr_fc1 0.002247340034867173 --lr_fc2 0.002247340034867173 --layer_norm 1 --dev 2 --hidden 256 --wd_sel 1.5258764793695603e-06 --lr_sel 0.005075721122049113 --step1_iter 100 --step2_iter 20 --max_feat_select 5 --num_adj 2

echo "====================="
echo "pokec"
echo "4-hop"
python -u node_class.py --data pokec --w_fc1 5.834619666270211e-07 --w_fc2 1.6585022017273566e-07 --w_fc3 1.6585022017273566e-07 --dropout1 0.9 --dropout2 0.6000000000000001 --dropout3 0.8 --lr_fc1 0.001917564105895921 --lr_fc2 0.001917564105895921 --layer_norm 1 --dev 7 --hidden 256 --wd_sel 1.9867889292382807e-05 --lr_sel 0.003241075172608557 --step1_iter 50 --step2_iter 20 --max_feat_select 5 --num_adj 2


echo "============"
echo "snap-patents"
echo "3-hops"
python -u node_class.py --data snap-patents --w_fc1 1.1588334706877353e-05 --w_fc2 0.00040131399456726143 --w_fc3 0.00040131399456726143 --dropout1 0.7000000000000001 --dropout2 0.6000000000000001 --dropout3 0.0 --lr_fc1 0.009519077545896818 --lr_fc2 0.009519077545896818 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 0.006272331166476561 --lr_sel 0.003852794948387357 --step1_iter 300 --step2_iter 20 --directed 1 --max_feat_select 10 --num_adj 4


echo "============"
echo "arxiv-year"
echo "3-hops"
python node_class.py --data arxiv-year --w_fc1 1.4804329350169762e-07 --w_fc2 8.134737804934632e-06 --w_fc3 8.134737804934632e-06 --dropout1 0.9 --dropout2 0.9 --dropout3 0.0 --lr_fc1 0.016116430573161755 --lr_fc2 0.016116430573161755 --layer_norm 1 --dev 1 --hidden 1024 --wd_sel 2.6106003825659324e-07 --lr_sel 0.008392327841098668 --step1_iter 300 --step2_iter 20 --directed 1 --max_feat_select 10 --num_adj 8

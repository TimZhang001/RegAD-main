conda activate pytorch


# 'bottle',    'cable', 'capsule', 'carpet', 'grid',       'hazelnut',   'leather', 
# 'metal_nut', 'pill',  'screw',   'tile',   'toothbrush', 'transistor', 'wood', 'zipper'
python train.py --obj bottle      --gpu 0
python train.py --obj cable       --gpu 0 
python train.py --obj capsule     --gpu 0

python train.py --obj carpet      --gpu 1
python train.py --obj grid        --gpu 1
python train.py --obj hazelnut    --gpu 1 

python train.py --obj leather     --gpu 2
python train.py --obj metal_nut   --gpu 2 
python train.py --obj pill        --gpu 2

python train.py --obj screw       --gpu 3
python train.py --obj tile        --gpu 3
python train.py --obj toothbrush  --gpu 3 

python train.py --obj transistor  --gpu 4
python train.py --obj wood        --gpu 4
python train.py --obj zipper      --gpu 4


python test.py --obj bottle      --gpu 0
python test.py --obj cable       --gpu 0 
python test.py --obj capsule     --gpu 0

python test.py --obj carpet      --gpu 1
python test.py --obj grid        --gpu 1
python test.py --obj hazelnut    --gpu 1 

python test.py --obj leather     --gpu 2
python test.py --obj metal_nut   --gpu 2 
python test.py --obj pill        --gpu 2

python test.py --obj screw       --gpu 3
python test.py --obj tile        --gpu 3
python test.py --obj toothbrush  --gpu 3 

python test.py --obj transistor  --gpu 4
python test.py --obj wood        --gpu 4
python test.py --obj zipper      --gpu 4
#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
		t) task=$OPTARG;;
        r) train="false";;
        p) predict="true";;
        
    esac
done
echo $name	
echo $train
echo $predict


# if ${train}
# then
	
# 	# cd /home/xychen/jsguo/nnFormer/nnformer/
#     cd /data/dulicui/project/git/nnFormer/nnformer/
# 	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_train 3d_fullres nnFormerTrainerV2_${name} ${task} 0
# fi

if ${predict}
then


	# cd /home/xychen/new_transformer/nnFormerFrame/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_ACDC/
    # cd /mnt/Data01/nnFormer/dataset/nnFormer_raw/nnFormer_raw_data/Task003_tumor/
    # cd /mnt/Data01/nnFormer/dataset/nnFormer_raw/nnFormer_raw_data/Task010_data00_cls6_test/
    cd /mnt/Data02/project/nnFormer/dataset/nnFormer_raw/nnFormer_raw_data/Task559_data00_cls6/
    # CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_final_checkpoint -tr nnFormerTrainerV2_${name}
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr nnFormerTrainerV2_${name} --disable_tta
	# python inference_acdc.py ${name}
    # python inference_tumor.py ${name}
fi




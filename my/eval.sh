#!/bin/bash

#export HF_HOME=/home/wangkai/wdtang/hr_infinity/huggingface
export HF_TOKEN="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"

export CUDA_VISIBLE_DEVICES=0

infer_eval_image_reward() {
    ${pip_ext} install image-reward pytorch_lightning
    ${pip_ext} install -U timm diffusers
    ${pip_ext} install openai==1.34.0 
    ${pip_ext} install httpx==0.23.0 
    #${pip_ext} install httpx==0.20.0 
    # step 1, infer images
    ${python_ext} /home/wangkai/wdtang/hr_infinity/text0/Infinity-main/evaluation/image_reward/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir  ${out_dir}

    # step 2, compute image reward
    
    #${pip_ext} install diffusers==0.16.0
    ${pip_ext} install diffusers==0.32.2
    #${pip_ext} install git+https://github.com/openai/CLIP.git ftfy
    ${pip_ext} install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/openai/CLIP.git ftfy
    ${python_ext} /home/wangkai/wdtang/hr_infinity/text0/Infinity-main/evaluation/image_reward/cal_imagereward.py \
    --meta_file ${out_dir}/metadata.jsonl
}

infer_eval_hpsv21() {
    ${pip_ext} install hpsv2
    ${pip_ext} install -U diffusers
    sudo apt install python3-tk
    wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
    mv bpe_simple_vocab_16e6.txt.gz /home/wangkai/wdtang/hr_infinity/hpsv21

    mkdir -p ${out_dir}
    ${python_ext} /home/wangkai/wdtang/hr_infinity/text0/Infinity-main/evaluation/hpsv2/eval_hpsv2.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images | tee ${out_dir}/log.txt
}

test_gen_eval() {
    #${pip_ext} install -U openmim
    #mim install mmengine mmcv-full==1.7.2
    #${pip_ext} install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0
    #${pip_ext} install -U diffusers
    #sudo apt install libgl1
    #${pip_ext} install openai==1.34.0
    #${pip_ext} install httpx==0.23.0

    # run inference
    #${python_ext} /home/wangkai/wdtang/hr_infinity/text0/Infinity-main/evaluation/gen_eval/infer4eval.py \
    ${python_ext} /root/zjm/Infinity/evaluation/gen_eval/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --rewrite_prompt ${rewrite_prompt}

    # detect objects
    #${python_ext} /home/wangkai/wdtang/hr_infinity/text0/Infinity-main/evaluation/gen_eval/evaluate_images.py ${out_dir}/images \
    ${python_ext} /root/zjm/Infinity/evaluation/gen_eval/evaluate_images.py ${out_dir}/images\
    --outfile ${out_dir}/results/det.jsonl \
    --model-config /root/zjm/Infinity/evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path /root/zjm/model

    # accumulate results
    #${python_ext} /home/wangkai/wdtang/hr_infinity/text0/Infinity-main/evaluation/gen_eval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    ${python_ext} /root/zjm/Infinity/evaluation/gen_eval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt
}

python_ext=python3
pip_ext=pip3

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=/root/zjm/model/infinity_2b_reg.pth
out_dir_root=/root/zjm/Infinity/out
vae_type=32
vae_path=/root/zjm/model/infinity_vae_d32reg.pth
cfg=4
tau=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=/root/.cache/huggingface/hub/models--google--flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}

# ImageReward
out_dir=${out_dir_root}/image_reward_${sub_fix}
#infer_eval_image_reward

# HPS v2.1
out_dir=${out_dir_root}/hpsv21_${sub_fix}
#infer_eval_hpsv21

# GenEval
rewrite_prompt=0
out_dir=${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_round2_real_rewrite
test_gen_eval


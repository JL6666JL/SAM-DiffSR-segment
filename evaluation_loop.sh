# python tasks/trainer.py \
# --config configs/sam/sam_diffsr_df2k4x.yaml \
# --benchmark \
# --hparams="test_save_png=True" \
# --exp_name sam_diffsr_df2k4x_caption_bs64 \
# --val_steps 364000 \
# --benchmark_name_list  test_Set5 test_Set14 test_Urban100 test_Manga109


# python tasks/trainer.py \
# --config configs/sam/sam_diffsr_df2k4x.yaml \
# --benchmark \
# --hparams="test_save_png=True" \
# --exp_name test \
# --val_steps 364000 \
# --benchmark_name_list  test_Set5 test_Set14 test_Urban100 test_Manga109


# python tasks/trainer.py \
# --config configs/sam/sam_diffsr_df2k4x.yaml \
# --benchmark_loop \
# --exp_name sam_diffsr_df2k4x_caption_bs64 \
# --work_dir /data1/jianglei/SAM-DiffSR/exp/result/ \
# --benchmark_name_list test_Set5 test_Set14 test_Urban100 test_Manga109 \
# --gt_img_path /data1/jianglei/SAM_testdata/data

python tasks/trainer.py \
--config configs/sam/sam_diffsr_df2k4x.yaml \
--benchmark_loop \
--exp_name test_loop \
--benchmark_name_list test_Set5 test_Set14 test_Urban100 test_Manga109 \

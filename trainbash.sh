# python -m  torch.distributed.launch tasks/trainer.py \
# --config configs/sam/sam_diffsr_df2k4x.yaml \
# --exp_name  sam_diffsr_df2k4x_caption \
# --reset \
# --hparams="rrdb_ckpt=weights/rrdb_div2k.ckpt" \
# --work_dir exp/
python  tasks/trainer.py \
--config configs/sam/sam_diffsr_df2k4x.yaml \
--exp_name test \
--reset \
--hparams="rrdb_ckpt=weights/rrdb_div2k.ckpt" \
--work_dir /data1/jianglei/SAM-DiffSR/exp
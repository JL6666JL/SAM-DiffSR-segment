python tasks/trainer.py \
--config configs/sam/sam_diffsr_df2k4x.yaml \
--exp_name test \
--reset \
--hparams="rrdb_ckpt=weights/rrdb_div2k.ckpt" \
--work_dir exp/
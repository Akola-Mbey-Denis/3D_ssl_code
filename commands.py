 python -m preprocess_ldis.lmdb_write --dataset scannet --split train --dump_root /projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_scannet/ --out /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/scannet_ldi_train_set.lmdb



python train_ssl.py  \
  --train_dataset "10_000 @ Hypersim_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --test_dataset  "1_000 @ Hypersim_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --model "vista_base()" \
  --lr 2e-4 \
  --epochs 100 \
  --batch_size 128 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vista_model_ssl_overfitting_model_100epochs/"



python train_sslv2.py  \
  --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --test_dataset  "Hypersim_NVS(split='val', overfit=True, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --model "vista_v2_base()" \
  --lr 2e-4 \
  --epochs 1000 \
  --batch_size 8 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 10 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vistav2_model_ssl_overfitting_model_1000epochs_two_losses/"




python train_ssl_baselines.py  \
  --train_dataset "10_000 @ Habitat_HM3D_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml')" \
  --test_dataset  "1_000 @ Habitat_HM3D_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml')" \
  --model "mae_large()" \
  --lr 2e-4 \
  --epochs 500 \
  --batch_size 64 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/mae_model_ssl_overfitting_model_training/"


python train_ssl_baselines.py  \
  --train_dataset "10_000 @ DL3DV_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/dl3dv_nvs.yaml')" \
  --test_dataset  "1_000 @ DL3DV_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/dl3dv_nvs.yaml')" \
  --model "mae_base()" \
  --lr 2e-4 \
  --epochs 500 \
  --batch_size 64 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/dl3dv_ns_mae_model_ssl_overfitting_model_training/"


python train_ssl_baselines.py  \
  --train_dataset "10_000 @ Hypersim_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --test_dataset  "1_000 @ Hypersim_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --model "mum_base(n_ref_views = 2, default_masking =False)" \
  --lr 2e-4 \
  --epochs 500 \
  --batch_size 64 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/hypersim_ns_mum_model_ssl_overfitting_model_training/"



python train_ssl_baselines.py  \
  --train_dataset "10_000 @ Replica_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/replica_nvs.yaml')" \
  --test_dataset  "1_000 @ Replica_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/replica_nvs.yaml')" \
  --model "mae_base()" \
  --lr 2e-4 \
  --epochs 500 \
  --batch_size 64 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/replica_nvs_mae_model_ssl_overfitting_model_training/"



python train_ssl_baselines.py  \
  --train_dataset "10_000 @ MegaDepth_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/megadepth_nvs.yaml')" \
  --test_dataset  "1_000 @ MegaDepth_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/megadepth_nvs.yaml')" \
  --model "mae_base()" \
  --lr 2e-4 \
  --epochs 500 \
  --batch_size 64 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/megadepth_nvs_mae_model_ssl_overfitting_model_training/"



python train_ssl_baselines.py  \
  --train_dataset "10_000 @ Taskonomy_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/taskonomy_nvs.yaml')" \
  --test_dataset  "1_000 @ Taskonomy_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/taskonomy_nvs.yaml')" \
  --model "mae_base()" \
  --lr 2e-4 \
  --epochs 500 \
  --batch_size 64 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/taskonomy_nvs_mae_model_ssl_overfitting_model_training/"




python train_ssl_muskie.py  \
--train_dataset "10_000 @ Hypersim_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
--test_dataset  "1_000 @ Hypersim_NVS(split='val', overfit=False, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
--model "muskie_base()" \
--lr 2e-4 \
--epochs 500 \
--batch_size 128 \
--accum_iter 1 \
--save_freq 10 \
--keep_freq 10 \
--eval_freq 1 \
--print_freq 1 \
--output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/muskie_base_model_ssl_overfitting_model_training/"



python train_ssl_baselines.py  \
  --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --test_dataset  "Hypersim_NVS(split='val', overfit=True, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --model "mum_base()" \
  --lr 2e-4 \
  --epochs 500 \
  --batch_size 32 \
  --accum_iter 2 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/mum_base_model_ssl_overfitting_model_training_6_views/"














python train_jepa.py \
  --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --test_dataset  "Hypersim_NVS(split='val', overfit=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --model "vista_jepa_base()" \
  --lr 1.5e-4 \
  --epochs 2000 \
  --batch_size 4 \
  --accum_iter 2 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --diag_freq 50 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vista_jepa_ssl_overfitting_2000_epochs/"


python train_ssl_jepa.py \
  --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --test_dataset  "Hypersim_NVS(split='val', overfit=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --model "vista_jepa_base()" \
  --lr 1.5e-4 \
  --epochs 500 \
  --batch_size 4 \
  --accum_iter 2 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 1 \
  --print_freq 1 \
  --diag_freq 50 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vista_jepa_ssl_overfitting_500_epochs/"

python train_stage2.py    --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')"   --test_dataset  "Hypersim_NVS(split='val', overfit=True, n_target_views=2, max_views =4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')"   --model "MuMCrossViewAutoEncoder_DPT()"   --lr 2e-4   --epochs 2000   --batch_size 2   --accum_iter 2   --save_freq 10   --keep_freq 10   --eval_freq 1   --print_freq 1   --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/stage_2_model_dpt/"


python train_stage2_depth_camera.py    --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')"   --test_dataset  "Hypersim_NVS(split='val', overfit=True, n_target_views=2, max_views =4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')"   --model "MuMCrossViewDecoders()"   --lr 2e-4   --epochs 1000   --batch_size 2   --accum_iter 2   --save_freq 10   --keep_freq 10   --eval_freq 1   --print_freq 1   --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/stage_2_model_dpt_camera_head/"


python finetune_ldi.py --train_dataset="ScannetPP_LDI(split='train', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml.yaml')" --test_dataset --train_dataset "ScannetPP_LDI(split='train', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml.yaml')" --ckpt_path="checkpoints/vista_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations"  --epochs=1000 --batch_size=8 --eval_freq=50 --save_freq=100 --keep_freq=50 --print_freq=1 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/LDI_overfit_experiment_Vista_L1_loss_scannet_ppp" 


python finetune_ldi.py --train_dataset="10_000 @ ScannetPP_LDI(split='train', overfit=False, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')" --test_dataset "ScannetPP_LDI(split='train', overfit=False, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')" --ckpt_path="checkpoints/vista_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations"  --epochs=1000 --batch_size=8 --eval_freq=50 --save_freq=100 --keep_freq=50 --print_freq=1 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/Vista_base_L1_loss_scannet_ppp"  --model "vista_base_ldi(num_layers=4)"


python finetune_ldi.py --train_dataset="10_000 @ ScanNet_LDI(split='train', overfit=False, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_ldi.yaml')" --test_dataset "ScanNet_LDI(split='val', overfit=False, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_ldi.yaml')" --ckpt_path="checkpoints/vista_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations" --epochs=70 --batch_size=128 --eval_freq=5 --save_freq=5 --keep_freq=5 --print_freq=10 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/Vista_base_L1_loss_scannetv2_ldi_prediction_model_new_trail_train" --model "vista_base_ldi(num_layers=4)"


 python -m preprocess_ldis.preprocess_ldi_lmdb --dataset scannet --split train --dump_root /projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_scannet/   --sequence_mode matrix  --overlap_basedir /projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/low_res_scannet_overlaps/ --overlap_scene_file  /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/scannet_low_res_overlap_scene_files.pkl  --out /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/scannet_train_ldi_data.lmdb



python finetune_ldi.py --train_dataset="ScannetPP_LDI(split='train', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')" --test_dataset "ScannetPP_LDI(split='val', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')" --ckpt_path="checkpoints/vista_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations"  --epochs=500 --batch_size=8 --eval_freq=10 --save_freq=5 --keep_freq=5 --print_freq=5 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/Vista_base_L1_loss_scannet_ppp_overfitting_model"  --model "vista_base_ldi(num_layers=4)"

python finetune_ldi.py --train_dataset="ScanNet_LDI(split='train', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_ldi.yaml')" --test_dataset "ScanNet_LDI(split='val', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_ldi.yaml')" --ckpt_path="checkpoints/vista_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations" --epochs=1000 --batch_size=8 --eval_freq=10 --save_freq=5 --keep_freq=5 --print_freq=5 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/Vista_base_L1_loss_scannetv2_overfitting_model" --model "vista_base_ldi(num_layers=4)"



python finetune_ldi.py --train_dataset="Matterport3D_LDI(split='train', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/matterport3d_ldi.yaml')" --test_dataset "Matterport3D_LDI(split='val', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/matterport3d_ldi.yaml')" --ckpt_path="checkpoints/vista_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations" --epochs=1000 --batch_size=8 --eval_freq=10 --save_freq=5 --keep_freq=5 --print_freq=5 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/Vista_base_L1_loss_matterport3d_overfitting_model" --model "vista_base_ldi(num_layers=4)"


python finetune_ldi.py --train_dataset="ARKitScenes_LDI(split='train', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/arkitscene_ldi.yaml')" --test_dataset "ARKitScenes_LDI(split='val', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/arkitscene_ldi.yaml')" --ckpt_path="checkpoints/vista_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations" --epochs=1000 --batch_size=8 --eval_freq=10 --save_freq=5 --keep_freq=5 --print_freq=5 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/Vista_base_L1_loss_arkitscenes_overfitting_model" --model "vista_base_ldi(num_layers=4)"


python finetune_ldi.py --train_dataset="ScanNet_LDI(split='train', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_ldi.yaml')" --test_dataset "ScanNet_LDI(split='val', overfit=True, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_ldi.yaml')"  --epochs=1000 --batch_size=8 --eval_freq=10 --save_freq=5 --keep_freq=5 --print_freq=5 --output_dir="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/poincar3_base_L1_loss_scannetv2_overfitting_model" --model "poincar3_ldi(num_layers=4)"




python train_sslv2.py  \
  --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --test_dataset  "Hypersim_NVS(split='val', overfit=True, n_target_views=2, max_views =6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" \
  --model "vista_v2_base(lambda_b=1.0, decoder_depth=2)" \ 
  --lr 2e-4 \
  --epochs 1000 \
  --batch_size 8 \
  --accum_iter 1 \
  --save_freq 10 \
  --keep_freq 10 \
  --eval_freq 10 \
  --print_freq 1 \
  --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vistav2_model_cross_view_decoder_depth_2_ssl_overfitting_model_1000epochs_two_losses/"

python train_nvd.py --train_dataset "Hypersim_NVS(split='train', overfit=True, n_target_views=1, max_views=3, img_size=224, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" --test_dataset "Hypersim_NVS(split='val', overfit=True, n_target_views=1, max_views=3, img_size=224, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" --model "vista_nvd_base_downstream()" --ckpt_path "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vista_v2_full_loss_b_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations/" --lr 2e-4 --epochs 1000 --batch_size 8 --accum_iter 1 --save_freq 10 --keep_freq 10 --eval_freq 10 --print_freq 1 --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/nvd_vista_overfit_test/"


python train_nvd.py --train_dataset "Hypersim_NVS(split='train', overfit=True, load_depth=True, n_target_views=1, max_views=3, img_size=224, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" --test_dataset "Hypersim_NVS(split='val', overfit=True, load_depth=True, n_target_views=1, max_views=3, img_size=224, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" --model "vista_nvd_base_downstream()" --ckpt_path "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vista_v2_full_loss_b_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations/"  --lr 2e-4 --epochs 3000 --batch_size 8 --accum_iter 1 --save_freq 100 --keep_freq 50 --eval_freq 50 --print_freq 10 --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/nvd_vista_overfit_testv4_plucker_tokenizer_vggt_loss_conf_loss/"



python train_nvd.py --train_dataset "Hypersim_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=6, img_size=224, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" --test_dataset "Hypersim_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=6, img_size=224, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" --model "vista_nvd_base_downstream()" --lr 2e-4 --epochs 40 --batch_size 256 --ckpt_path "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vista_v2_full_loss_b_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations/"  --lr 2e-4 --epochs 40 --batch_size 128 --accum_iter 1 --save_freq 5 --keep_freq 5 --eval_freq 5 --print_freq 100 --output_dir "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/nvd_vista_finetune_hypersim_testv4_plucker_tokenizer_vggt_depth_loss/"

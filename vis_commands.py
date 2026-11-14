#NVD visualizations

python -m visualizations.visualize_nvd --model_name vista  --model vista_basev2_downstream--ckpt vista_base_model_nvd_probe_all_datasets_scannetpp_model_224_x_224_30ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --num_samples 40 --out_dir results/figures/nvd_scannetpp_vis --num_workers 4

python -m visualizations.visualize_nvd --model_name mum  --model vista_basev2_downstream --ckpt vista_base_model_nvd_probe_all_datasets_scannetpp_model_224_x_224_30ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --num_samples 40 --out_dir results/figures/nvd_scannetpp_vis --num_workers 4


# 3D_SLL depths and point clouds visualiztions


python -m visualizations.visualize_ssl --model_name vista --model vista_basev2_downstream --ckpt vistav2_full_weight_b_c_heads_scene_norm_patch_tokens_pose_base_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep_latest --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4

python -m visualizations.visualize_ssl --model_name muskie --model muskie_base_downstream --ckpt muskie_scene_norm_base_default_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4

python -m visualizations.visualize_ssl --model_name mae --model mae_base_downstream --ckpt mae_base_scene_norm_patch_tokens_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4

python -m visualizations.visualize_ssl --model_name mum --model mum_base_downstream --ckpt mum_base_scene_norm_patch_tokens_default_random_masking_75_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4





python -m visualizations.visualize_ssl --model_name vista --model vista_basev2_downstream --ckpt vistav2_full_weight_b_c_heads_scene_norm_patch_tokens_pose_base_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep_latest --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4

python -m visualizations.visualize_ssl --model_name muskie --model muskie_base_downstream --ckpt muskie_scene_norm_base_default_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4

python -m visualizations.visualize_ssl --model_name mae --model mae_base_downstream --ckpt mae_base_scene_norm_patch_tokens_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4

python -m visualizations.visualize_ssl --model_name mum --model mum_base_downstream --ckpt mum_base_scene_norm_patch_tokens_default_random_masking_75_model_downstream_depth_camera_all_datasets_model_224_x_224_40ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_nrgbd_vis" --num_workers 4




python -m visualizations.visualize_ldis --model_name vista --model "vista_base_ldi(num_layers=4)" --ckpt vista_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_only_bs64_model_224_x_224_140ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/ldi_scannetpp_vis" --num_workers 4

python -m visualizations.visualize_ldis --model_name muskie --model "muskie_base_ldi(num_layers=4)" --ckpt muskie_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_bs64_correct_model_224_x_224_140ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/ldi_scannetpp_vis" --num_workers 4

python -m visualizations.visualize_ldis --model_name mae --model "mae_base_ldi(num_layers=4)" --ckpt mae_ldi_scannetpp_140ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/ldi_scannetpp_vis" --num_workers 4

python -m visualizations.visualize_ldis --model_name mum --model "mum_base_ldi(num_layers=4)" --ckpt mum_ldi_scannetpp_140ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/ldi_scannetpp_vis" --num_workers 4

python -m visualizations.visualize_ldis --model_name poincar3 --model "vista_base_ldi(num_layers=4)" --ckpt mum_ldi_scannetpp_140ep --dataset "NRGBD_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/ldi_scannetpp_vis" --num_workers 4




python -m visualizations.visualize_ldi --model_name vista --model "vista_base_ldi(num_layers=4)" --ckpt vista_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_max_views_4_bs64_model_224_x_224_140ep --dataset "ScannetPP_LDI(split='val', max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')" --indices 0,12,40 --out_dir "./cvpr2027_results_visuals/figures/ldi_scannetpp_vis" --num_workers 4






python debug_gt_poses.py --dataset "Habitat_HM3D_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --n 3

python -m visualizations.visualize_ssl --model_name muskie --model muskie_base_downstream --ckpt muskie_base_default_model_downstream_depth_camera_all_datasets_model_224_x_224_70ep --dataset "Habitat_HM3D_NVS(split='vis', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml', img_size=224, max_views=4, n_target_views=2, load_depth=True)" --out_dir "./cvpr2027_results_visuals/figures/3d_ssl_depth_pc_habitat_vis" --num_workers 4


python -m visualizations.visualize_ssl --model_name vista_overfit --model vista_basev2_downstream --ckpt vistav2_downstream_overfit100_2000ep --dataset "Hypersim_NVS(split='train', overfit=True, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml')" --out_dir "./cvpr2027_results_visuals/figures/overfit_check" --num_workers 4
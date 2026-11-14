
###LDI evaluation
python -m evaluation.eval_ldis --model "vista_base_ldi(num_layers=4)" --model_name vista --ckpt vista_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_bs64_model_224_x_224_140ep --dataset "NRGBD_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"
python -m evaluation.eval_ldis --model "vista_base_ldi(num_layers=4)" --model_name vista --ckpt vista_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_bs64_model_224_x_224_140ep --dataset "ScannetPP_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"




python -m evaluation.eval_ldis --model "muskie_base_ldi(num_layers=4)" --model_name muskie --ckpt muskie_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_bs64_correct_model_224_x_224_140ep --dataset "NRGBD_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"
python -m evaluation.eval_ldis --model "muskie_base_ldi(num_layers=4)" --model_name muskie --ckpt muskie_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_bs64_correct_model_224_x_224_140ep --dataset "ScannetPP_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"




python -m evaluation.eval_ldis --model "mae_base_ldi(num_layers=4)" --model_name mae --ckpt mae_ldi_scannetpp_140ep --dataset "NRGBD_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"
python -m evaluation.eval_ldis --model "mae_base_ldi(num_layers=4)" --model_name mae --ckpt mae_ldi_scannetpp_140ep --dataset "ScannetPP_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"



python -m evaluation.eval_ldis --model "mum_base_ldi(num_layers=4)" --model_name mum --ckpt mum_ldi_scannetpp_140ep --dataset "NRGBD_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/nrgbd_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"
python -m evaluation.eval_ldis --model "mum_base_ldi(num_layers=4)" --model_name mum --ckpt mum_ldi_scannetpp_140ep --dataset "ScannetPP_LDI(split='test', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')"  --results_dir "./cvpr_2027_ssl_results_main_ldis/"

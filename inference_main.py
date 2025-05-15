import logging

import soundfile
import torch
import fairseq.data

from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # Mandatory parameters
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_37600.pth", help='Model path')
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json", help='Configuration file path')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='Force audio slicing. Default is 0 for auto slicing. Unit: seconds')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='List of wav file names, should be placed in the raw folder')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='Pitch adjustment, supports both positive and negative values (in semitones)')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['buyizi'], help='Name(s) of target speakers for synthesis')
    
    # Optional parameters
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='Automatically predict F0 for voice conversion. Do NOT enable this when converting singing, as it will cause out-of-tune results.')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="", help='Path to clustering model or feature retrieval index. Leave empty to automatically use the default path for each model. If clustering or feature retrieval is not trained, fill this arbitrarily.')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='Clustering scheme or feature retrieval ratio, range 0-1. Default to 0 if clustering or feature retrieval model not trained.')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='Crossfade length for two audio slices. Adjust if forced slicing causes vocal discontinuity. If smooth, recommend using the default value of 0. Unit: seconds.')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="pm", help='Choose F0 predictor: options are crepe, pm, dio, harvest, rmvpe, fcpe. Default is pm. (Note: crepe uses mean filter for original F0)')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='Use NSF_HIFIGAN enhancer. Can improve audio quality for models with small training sets, but may have negative effects on well-trained models. Disabled by default.')
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=False, help='Enable shallow diffusion to solve some artifacts/electronic sounds. Disabled by default. When enabled, NSF_HIFIGAN enhancer is disabled.')
    parser.add_argument('-usm', '--use_spk_mix', action='store_true', default=False, help='Whether to use speaker blending (character fusion)')
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=1, help='Blending ratio for replacing source loudness envelope with output loudness envelope; closer to 1 means more of output envelope is used')
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', default=False, help='Enable feature retrieval. If enabled, clustering model will be disabled, and parameters cm and cr become the index path and mixing ratio for feature retrieval.')

    # Shallow diffusion settings
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="logs/44k/diffusion/model_0.pt", help='Diffusion model path')
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="logs/44k/diffusion/config.yaml", help='Diffusion model configuration file path')
    parser.add_argument('-ks', '--k_step', type=int, default=100, help='Number of diffusion steps. Higher values produce results closer to the diffusion model. Default is 100.')
    parser.add_argument('-se', '--second_encoding', action='store_true', default=False, help='Double encoding: encodes the original audio a second time before shallow diffusion. An experimental (luck-based) option: sometimes improves, sometimes worsens results.')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='Diffusion-only mode. Sovits model will not be loaded; only the diffusion model will be used for inference.')
    

    # Parameters that usually don't need to be changed
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='Default is -40. For noisy audio, set to -30; for dry vocals with breathing, set to -50.')
    parser.add_argument('-d', '--device', type=str, default=None, help='Inference device. Set to None to automatically select between CPU and GPU.')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='Noise level. Affects articulation and audio quality. Somewhat experimental.')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='Padding seconds for inference audio. Due to unknown reasons, artifacts may appear at the start and end; adding a short silent pad solves this.')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav', help='Output audio format')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='After automatic audio slicing, the head and tail of each slice are discarded. This parameter sets the proportion of crossfade length to retain, range 0-1 (left-open, right-closed interval)')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0, help='Allows the enhancer to adapt to higher pitch ranges (in semitones). Default is 0.')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,help='F0 filtering threshold. Only effective when using crepe. Value range: 0-1. Lowering this value reduces pitch drift probability, but increases muting.')


    args = parser.parse_args()

    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    f0p = args.f0_predictor
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold
    diffusion_model_path = args.diffusion_model_path
    diffusion_config_path = args.diffusion_config_path
    k_step = args.k_step
    only_diffusion = args.only_diffusion
    shallow_diffusion = args.shallow_diffusion
    use_spk_mix = args.use_spk_mix
    second_encoding = args.second_encoding
    loudness_envelope_adjustment = args.loudness_envelope_adjustment

    if cluster_infer_ratio != 0:
        if args.cluster_model_path == "":
            if args.feature_retrieval:  # 若指定了占比但没有指定模型路径，则按是否使用特征检索分配默认的模型路径
                args.cluster_model_path = "logs/44k/feature_and_index.pkl"
            else:
                args.cluster_model_path = "logs/44k/kmeans_10000.pt"
    else:  # 若未指定占比，则无论是否指定模型路径，都将其置空以避免之后的模型加载
        args.cluster_model_path = ""

    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device,
                    args.cluster_model_path,
                    enhance,
                    diffusion_model_path,
                    diffusion_config_path,
                    shallow_diffusion,
                    only_diffusion,
                    use_spk_mix,
                    args.feature_retrieval)
    
    infer_tool.mkdir(["raw", "results"])
    
    if len(spk_mix_map)<=1:
        use_spk_mix = False
    if use_spk_mix:
        spk_list = [spk_mix_map]
    
    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,
                "cluster_infer_ratio" : cluster_infer_ratio,
                "auto_predict_f0" : auto_predict_f0,
                "noice_scale" : noice_scale,
                "pad_seconds" : pad_seconds,
                "clip_seconds" : clip,
                "lg_num": lg,
                "lgr_num" : lgr,
                "f0_predictor" : f0p,
                "enhancer_adaptive_key" : enhancer_adaptive_key,
                "cr_threshold" : cr_threshold,
                "k_step":k_step,
                "use_spk_mix":use_spk_mix,
                "second_encoding":second_encoding,
                "loudness_envelope_adjustment":loudness_envelope_adjustment
            }
            audio = svc_model.slice_inference(**kwarg)
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            isdiffusion = "sovits"
            if shallow_diffusion :
                isdiffusion = "sovdiff"
            if only_diffusion :
                isdiffusion = "diff"
            if use_spk_mix:
                spk = "spk_mix"
            res_path = f'results/{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()

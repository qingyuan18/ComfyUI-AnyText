import os
import folder_paths
import torch
import node_helpers
from PIL import Image, ImageOps, ImageSequence
import hashlib
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
comfyui_models_dir = folder_paths.models_dir
temp_txt_path = os.path.join(current_directory, "temp_dir", "AnyText_temp.txt")
temp_img_path = os.path.join(current_directory, "temp_dir", "AnyText_manual_mask_pos_img.png")

class AnyText_loader:
    @classmethod
    def INPUT_TYPES(cls):
        font_list = os.listdir(os.path.join(comfyui_models_dir, "fonts"))
        checkpoints_list = folder_paths.get_filename_list("checkpoints")
        clip_list = os.listdir(os.path.join(comfyui_models_dir, "clip"))
        translator_list = os.listdir(os.path.join(comfyui_models_dir, "prompt_generator"))
        font_list.insert(0, "Auto_DownLoad")
        checkpoints_list.insert(0, "Auto_DownLoad")
        clip_list.insert(0, "Auto_DownLoad")
        translator_list.insert(0, "Auto_DownLoad")

        return {
            "required": {
                "font": (font_list, ),
                "ckpt_name": (checkpoints_list, ),
                "clip": (clip_list, ),
                "translator": (translator_list, ),
                # "show_debug": ("BOOLEAN", {"default": False}),
                }
            }

    RETURN_TYPES = ("AnyText_Loader", )
    RETURN_NAMES = ("AnyText_Loader", )
    FUNCTION = "AnyText_loader_fn"
    CATEGORY = "ExtraModels/AnyText"
    TITLE = "AnyText Loader"

    def AnyText_loader_fn(self, 
                          font, 
                          ckpt_name, 
                          clip, 
                          translator, 
                        #   show_debug
                          ):
        font_path = os.path.join(comfyui_models_dir, "fonts", font)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_path = os.path.join(current_directory, 'models_yaml', 'anytext_sd15.yaml')
        if clip != 'Auto_DownLoad':
                clip_path = os.path.join(comfyui_models_dir, "clip", clip)
        else:
                clip_path = clip
        if translator != 'Auto_DownLoad':
                translator_path = os.path.join(comfyui_models_dir, "prompt_generator", translator)
        else:
                translator_path = translator
        
        #将输入参数合并到一个参数里面传递到.nodes
        loader = (font_path + "|" + str(ckpt_path) + "|" + clip_path + "|" + translator_path + "|" + cfg_path)
        
        # if show_debug == True:
        #     print(f'\033[93mloader(合并后的4个输入参数，传递给nodes): {loader} \033[0m\n \
        #             \033[93mfont_path(字体): {font_path} \033[0m\n \
        #             \033[93mckpt_path(AnyText模型): {ckpt_path} \033[0m\n \
        #             \033[93mclip_path(clip模型): {clip_path} \033[0m\n \
        #             \033[93mtranslator_path(翻译模型): {translator_path} \033[0m\n \
        #             \033[93myaml_file(yaml配置文件): {cfg_path} \033[0m\n')
        return (loader, )

class AnyText_Pose_IMG:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                        "image": (sorted(files), {"image_upload": True}),
                        },
                }

    CATEGORY = "ExtraModels/AnyText"
    RETURN_TYPES = (
        "AnyText_images", 
        "IMAGE")
    RETURN_NAMES = (
        "AnyText_images", 
        "mask_img")
    FUNCTION = "AnyText_Pose_IMG"
    TITLE = "AnyText Pose IMG"
    
    def AnyText_Pose_IMG(self, image):
        ori_image_path = folder_paths.get_annotated_filepath(image)
        pos_img_path = os.path.join(temp_img_path)
        AnyText_images = ori_image_path + '|' + pos_img_path
        img = node_helpers.pillow(Image.open, ori_image_path)
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            # output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            # output_image = output_images[0]
            output_mask = output_masks[0]
        invert_mask = 1.0 - output_mask
        inverted_mask_image = invert_mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        i = 255. * inverted_mask_image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(temp_img_path)

        return (
            AnyText_images, 
            inverted_mask_image)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

from modelscope.utils.constant import Tasks
class AnyText_translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (['cpu', 'gpu'] , {"default": "gpu"}),
                "prompt": ("STRING", {"default": "这里是单批次翻译文本输入。\n声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。\n尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！\n使用换行，效果可能更佳。", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "if_Batch": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("中译英结果",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "AnyText_translator"
    TITLE = "AnyText中译英-阿里达摩院damo/nlp_csanmt_translation_zh2en"

    def AnyText_translator(self, prompt, Batch_prompt, if_Batch, device, Batch_Newline):
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        if if_Batch == True:
            input_sequence = Batch_prompt
            # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
            input_sequence = '<SENT_SPLIT>'.join(input_sequence)
        else:
            input_sequence = prompt
        if os.access(os.path.join(comfyui_models_dir, "prompt_generator", "nlp_csanmt_translation_zh2en", "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
            zh2en_path = os.path.join(comfyui_models_dir, 'prompt_generator', 'nlp_csanmt_translation_zh2en')
        else:
            zh2en_path = "damo/nlp_csanmt_translation_zh2en"
        
        if not is_module_imported('pipeline'):
            from modelscope.pipelines import pipeline
        pipeline_ins = pipeline(task=Tasks.translation, model=zh2en_path, device=device)
        outputs = pipeline_ins(input=input_sequence)
        if if_Batch == True:
            results = outputs['translation'].split('<SENT_SPLIT>')
            if Batch_Newline == True:
                results = '\n\n'.join(results)
            else:
                results = ' '.join(results)
        else:
            results = outputs['translation']
        with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(results)
        return (results, )

def is_module_imported(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def is_folder_exist(folder_path):
    result = os.path.exists(folder_path)
    return result

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyText_loader": AnyText_loader,
    "AnyText_Pose_IMG": AnyText_Pose_IMG,
    "AnyText_translator": AnyText_translator,
}
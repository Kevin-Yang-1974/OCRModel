from tqdm import tqdm
import json
import argparse
from transformers import AutoTokenizer
import torch
import os
import uuid               # [新增] 用于生成唯一ID
from pathlib import Path  # [新增] 用于路径操作
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria
from PIL import Image
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image

output_list = []

def eval_model(args):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(model_name, 
                                               low_cpu_mem_usage=True, 
                                               device_map='cuda', 
                                               use_safetensors=True, 
                                               pad_token_id=151643,
                                            ).eval()
    model.to(device='cuda')
    gts_path = args.gtfile_path
    gts = json.load(open(gts_path))  
    image_processor = BlipImageEvalProcessor(image_size=1024)

    print("Generate Results......")
    for ann in tqdm(gts):
        output_json = {}
        image_file = ann["image"] 
        image_file_path = os.path.join(args.image_path, image_file) # 命令行和json内路径拼接
        image = load_image(image_file_path)
        image_token_len = 256
        qs = 'OCR: '
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
        conv_mode = "mpt"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        image_1 = image.copy()
        image_tensor = image_processor(image)
        image_tensor_1 = image_processor(image_1)
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(), image_tensor_1.unsqueeze(0).to(dtype=torch.bfloat16).cuda())],
            do_sample=False,
            num_beams = 1,
            no_repeat_ngram_size = 20,
            # streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:],skip_special_tokens=True).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        output_json['image'] = ann["image"]
        output_json['question'] = prompt 
        output_json['label'] = ann["conversations"][1]["value"]
        output_json['answer'] = outputs
        output_list.append(output_json)

    filename = args.out_file
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir,exist_ok=True)
    with open(filename, 'w', encoding="utf-8") as file_obj:
        json.dump(output_list, file_obj, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--gtfile_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="./results_final.json")
    args = parser.parse_args()
    
    #Params check
    if not args.model_name:
        raise ValueError("model-name is required")

    # ================= 核心修改开始 =================
    # 1. 获取原始路径
    original_model_path = Path(args.model_name).resolve()
    
    # 2. 生成一个符合 Python 变量命名规范的临时名字（无点号，无横杠）
    # 使用 uuid 生成唯一标识，避免多任务运行时冲突
    clean_dir_name = f"temp_got_{uuid.uuid4().hex}"
    
    # 3. 确定软链的位置（建议放在原模型目录的父级，或者同级）
    # 这里放在同级目录下
    symlink_path = original_model_path.parent / clean_dir_name

    print(f"🔄 Creating symlink: {symlink_path} -> {original_model_path}")
    
    try:
        # 创建软链接
        if not symlink_path.exists():
            os.symlink(original_model_path, symlink_path)
        
        # 4. 【关键步骤】欺骗 eval_model 使用这个新路径
        # 这样 transformers 加载时看到的包名就是 temp_got_xxxx，合法且无冲突
        args.model_name = str(symlink_path)
        
        print(args)
        eval_model(args)

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise e # 抛出异常以便调试

    finally:
        # 5. 清理现场：无论运行成功还是报错，都删除软链接
        if symlink_path.exists() and symlink_path.is_symlink():
            print(f"🧹 Cleaning up symlink: {symlink_path}")
            try:
                os.unlink(symlink_path)
            except Exception as cleanup_err:
                print(f"⚠️ Failed to remove symlink: {cleanup_err}")
    # ================= 核心修改结束 =================
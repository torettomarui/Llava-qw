import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import os

import torch
from torch.utils.data import Dataset

from PIL import Image


import transformers
import tokenizers
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from Models.conversation import get_conv_template

import dataclasses
from enum import IntEnum, auto
from typing import Any, Dict, List, Tuple, Union



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = '<|vision_pad|>'
IMG_START_TOKEN = '<|vision_start|>'
IMG_END_TOKEN = '<|vision_end|>'


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def image_stack(batch):
    img_tensor_list = []
    for k in batch:
        for img_file in k['pixel_values']:
            img = Image.open(img_file).convert('RGB')
            img_tensor_list += [img]
    transform = build_transform(448)
    pixel_values = [transform(image) for image in img_tensor_list]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def LlavaQwDataCollateFn(batch):

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f['input_ids']) for f in batch],
        padding_value=151643,
        batch_first=True
    )

    pixel_values = image_stack(batch)

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f['attention_mask']) for f in batch],
        padding_value=0,
        batch_first=True
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f['labels']) for f in batch],
        padding_value=-100,
        batch_first=True
    )

    return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


excluded_image = ["ocr_vqa/images/835607240.jpg",
"ocr_vqa/images/71351817.jpg",
"ocr_vqa/images/014005667X.jpg",
"ocr_vqa/images/471148288.jpg",
"ocr_vqa/images/134412052.jpg",
"ocr_vqa/images/1572240466.jpg",
"ocr_vqa/images/750219378.jpg",
"ocr_vqa/images/1560449152.jpg",
"ocr_vqa/images/1570761493.jpg",
"ocr_vqa/images/1562613480.jpg",
"ocr_vqa/images/521506743.jpg",
"ocr_vqa/images/739704133.jpg",
"ocr_vqa/images/1565540581.jpg",
"ocr_vqa/images/060960323X.jpg",
"ocr_vqa/images/786405511.jpg",
"ocr_vqa/images/1556226535.jpg",
"ocr_vqa/images/671025627.jpg",
"ocr_vqa/images/892133252.jpg",
"ocr_vqa/images/039309071X.jpg",
"ocr_vqa/images/1573440906.jpg",
"ocr_vqa/images/673384772.jpg",
"ocr_vqa/images/2067009559.jpg",
"ocr_vqa/images/377570177X.jpg",
"ocr_vqa/images/915801841.jpg",
"ocr_vqa/images/156347185X.jpg",
"ocr_vqa/images/472084798.jpg",
"ocr_vqa/images/789408554.jpg",
"ocr_vqa/images/882894293.jpg",
"ocr_vqa/images/965776611.jpg",
"ocr_vqa/images/155583468X.jpg",
"ocr_vqa/images/1559920696.jpg",
"ocr_vqa/images/292796048.jpg",
"ocr_vqa/images/231072430.jpg",
"ocr_vqa/images/679844023.jpg",
"ocr_vqa/images/876043082.jpg",
"ocr_vqa/images/080740604X.jpg",
"ocr_vqa/images/157851231X.jpg",
"ocr_vqa/images/1881320200.jpg",
"ocr_vqa/images/1571971459.jpg",
"ocr_vqa/images/877793417.jpg",
"ocr_vqa/images/967695104.jpg",
"ocr_vqa/images/878576991.jpg",
"ocr_vqa/images/674035275.jpg",
"ocr_vqa/images/1566864941.jpg",
"ocr_vqa/images/893148423.jpg",
"ocr_vqa/images/078942049X.jpg",
"ocr_vqa/images/B001KBBD4A.jpg",
"ocr_vqa/images/1887089160.jpg",
"ocr_vqa/images/785282394.jpg",
"ocr_vqa/images/739705539.jpg",
"ocr_vqa/images/1576730867.jpg",
"ocr_vqa/images/878918515.jpg",
"ocr_vqa/images/067944680X.jpg",
"ocr_vqa/images/1564583015.jpg",
"ocr_vqa/images/1862045852.jpg",
"ocr_vqa/images/312244452.jpg",
"ocr_vqa/images/B005FOFNA8.jpg",
"ocr_vqa/images/375400664.jpg",
"ocr_vqa/images/939302322.jpg",
"ocr_vqa/images/933261004.jpg",
"ocr_vqa/images/1560799722.jpg",
"ocr_vqa/images/031476271X.jpg",
"ocr_vqa/images/912818034.jpg",
"ocr_vqa/images/087033512X.jpg",
"ocr_vqa/images/1564965112.jpg",
"ocr_vqa/images/B00XIZWWNC.jpg",
"ocr_vqa/images/679888268.jpg",
"ocr_vqa/images/898620996.jpg",
"ocr_vqa/images/739715100.jpg",
"ocr_vqa/images/810943794.jpg",
"ocr_vqa/images/898796482.jpg",
"ocr_vqa/images/B013RVJ7KW.jpg",
"ocr_vqa/images/531202852.jpg",
"ocr_vqa/images/739702505.jpg",
"ocr_vqa/images/1885928017.jpg",
"ocr_vqa/images/1878239376.jpg",
"ocr_vqa/images/345351452.jpg",
"ocr_vqa/images/739714023.jpg",
"ocr_vqa/images/64462013.jpg",
"ocr_vqa/images/375501983.jpg",
"ocr_vqa/images/391040952.jpg",
"ocr_vqa/images/70359148.jpg",
"ocr_vqa/images/816512019.jpg",
"ocr_vqa/images/960536205.jpg",
"ocr_vqa/images/412132710.jpg",
"ocr_vqa/images/B011M9LHUO.jpg",
"ocr_vqa/images/471550833.jpg",
"ocr_vqa/images/671766627.jpg",
"ocr_vqa/images/1564582922.jpg",
"ocr_vqa/images/968557902.jpg",
"ocr_vqa/images/679445765.jpg",
"ocr_vqa/images/877792356.jpg",
"ocr_vqa/images/3928819232.jpg",
"ocr_vqa/images/1890838004.jpg",
"ocr_vqa/images/936783109.jpg",
"ocr_vqa/images/553062204.jpg",
"ocr_vqa/images/870675656.jpg",
"ocr_vqa/images/395539331.jpg",
"ocr_vqa/images/715308904.jpg",
"ocr_vqa/images/870408712.jpg",
"ocr_vqa/images/942110439.jpg",
"ocr_vqa/images/1560443952.jpg",
"ocr_vqa/images/28608194.jpg",
"ocr_vqa/images/268016860.jpg",
"ocr_vqa/images/1561701289.jpg",
"ocr_vqa/images/155850835X.jpg",
"ocr_vqa/images/1559920734.jpg",
"ocr_vqa/images/934710171.jpg",
"ocr_vqa/images/393090027.jpg",
"ocr_vqa/images/737303360.jpg",
"ocr_vqa/images/891960856.jpg",
"ocr_vqa/images/520081854.jpg",
"ocr_vqa/images/832904651.jpg",
"ocr_vqa/images/1566250420.jpg",
"ocr_vqa/images/818405988.jpg",
"ocr_vqa/images/809235269.jpg",
"ocr_vqa/images/006270110X.jpg",
"ocr_vqa/images/812035712.jpg",
"ocr_vqa/images/067173363X.jpg",
"ocr_vqa/images/892390263.jpg",
"ocr_vqa/images/893463264.jpg",
"ocr_vqa/images/1855326779.jpg",
"ocr_vqa/images/843129697.jpg",
"ocr_vqa/images/961742607.jpg",
"ocr_vqa/images/295968265.jpg",
"ocr_vqa/images/1563705176.jpg",
"ocr_vqa/images/553377833.jpg",
"ocr_vqa/images/78821185.jpg",
"ocr_vqa/images/393098745.jpg",
"ocr_vqa/images/553069403.jpg",
"ocr_vqa/images/1580910068.jpg",
"ocr_vqa/images/2061514022.jpg",
"ocr_vqa/images/412076217.jpg",
"ocr_vqa/images/807085707.jpg",
"ocr_vqa/images/968297072.jpg",
"ocr_vqa/images/312890583.jpg",
"ocr_vqa/images/B007K53FQ4.jpg",
"ocr_vqa/images/1570611912.jpg",
"ocr_vqa/images/935926194.jpg",
"ocr_vqa/images/688116191.jpg",
"ocr_vqa/images/1579771009.jpg",
"ocr_vqa/images/139642625.jpg",
"ocr_vqa/images/393314286.jpg",
"ocr_vqa/images/3980621154.jpg",
"ocr_vqa/images/1555951120.jpg",
"ocr_vqa/images/761525904.jpg",
"ocr_vqa/images/966355903.jpg",
"ocr_vqa/images/345414810.jpg",
"ocr_vqa/images/1569750882.jpg",
"ocr_vqa/images/471178705.jpg",
"ocr_vqa/images/958315434.jpg",
"ocr_vqa/images/1568362021.jpg",
"ocr_vqa/images/830816011.jpg",
"ocr_vqa/images/821411896.jpg",
"ocr_vqa/images/517884283.jpg",
"ocr_vqa/images/898796490.jpg",
"ocr_vqa/images/1886947694.jpg",
"ocr_vqa/images/415913756.jpg",
"ocr_vqa/images/1881174034.jpg",
"ocr_vqa/images/093062596X.jpg",
"ocr_vqa/images/761522751.jpg",
"ocr_vqa/images/812015320.jpg",
"ocr_vqa/images/078710339X.jpg",
"ocr_vqa/images/810940183.jpg",
"ocr_vqa/images/739704516.jpg",
"ocr_vqa/images/739715534.jpg",
"ocr_vqa/images/60164255.jpg",
"ocr_vqa/images/810955563.jpg",
"ocr_vqa/images/914625179.jpg",
"ocr_vqa/images/3884452762.jpg",
"ocr_vqa/images/073970477X.jpg",
"ocr_vqa/images/962726303.jpg",
"ocr_vqa/images/688151175.jpg",
"ocr_vqa/images/967008808.jpg",
"ocr_vqa/images/811820580.jpg",
"ocr_vqa/images/962770124.jpg",
"ocr_vqa/images/786884061.jpg",
"ocr_vqa/images/812931432.jpg",
"ocr_vqa/images/030724055X.jpg",
"ocr_vqa/images/471542989.jpg",
"ocr_vqa/images/1565042301.jpg",
"ocr_vqa/images/875730434.jpg",
"ocr_vqa/images/939302349.jpg",
"ocr_vqa/images/1564588963.jpg",
"ocr_vqa/images/1564556409.jpg",
"ocr_vqa/images/201112973.jpg",
"ocr_vqa/images/931580390.jpg",
"ocr_vqa/images/912111364.jpg",
"ocr_vqa/images/739714600.jpg",
"ocr_vqa/images/750223391.jpg",
"ocr_vqa/images/28628594.jpg",
"ocr_vqa/images/789404427.jpg",
"ocr_vqa/images/3980621146.jpg",
"ocr_vqa/images/761511253.jpg",
"ocr_vqa/images/4544040604.jpg",
"ocr_vqa/images/28624084.jpg",
"ocr_vqa/images/289800900.jpg",
"ocr_vqa/images/1574320793.jpg",
"ocr_vqa/images/840734921.jpg",
"ocr_vqa/images/1885726031.jpg",
"ocr_vqa/images/1557987203.jpg",
"ocr_vqa/images/670886939.jpg",
"ocr_vqa/images/1562614479.jpg",
"ocr_vqa/images/446387355.jpg",
"ocr_vqa/images/812903390.jpg",
"ocr_vqa/images/1580170536.jpg",
"ocr_vqa/images/763115509.jpg",
"ocr_vqa/images/679441662.jpg",
"ocr_vqa/images/870695916.jpg",
"ocr_vqa/images/879725109.jpg",
"ocr_vqa/images/860208656.jpg",
"ocr_vqa/images/761500413.jpg",
"ocr_vqa/images/205260780.jpg",
"ocr_vqa/images/1878239589.jpg",
"ocr_vqa/images/706377273.jpg",
"ocr_vqa/images/749517735.jpg",
"ocr_vqa/images/038794740X.jpg",
"ocr_vqa/images/1574301012.jpg",
"ocr_vqa/images/452255260.jpg",
"ocr_vqa/images/1555603513.jpg",
"ocr_vqa/images/133099156.jpg",
"ocr_vqa/images/B00WTKH3HC.jpg",
"ocr_vqa/images/185230863X.jpg",
"ocr_vqa/images/689852649.jpg",
"ocr_vqa/images/1575662698.jpg",
"ocr_vqa/images/093727464X.jpg",
"ocr_vqa/images/965150739.jpg",
"ocr_vqa/images/1558216103.jpg",
"ocr_vqa/images/425099369.jpg",
"ocr_vqa/images/471024961.jpg",
"ocr_vqa/images/471523771.jpg",
"ocr_vqa/images/1580621783.jpg",
"ocr_vqa/images/081182568X.jpg",
"ocr_vqa/images/055305340X.jpg",
"ocr_vqa/images/553353500.jpg",
"ocr_vqa/images/671567918.jpg",
"ocr_vqa/images/1565047168.jpg",
"ocr_vqa/images/810928949.jpg",
"ocr_vqa/images/1579901387.jpg",
"ocr_vqa/images/930031571.jpg",
"ocr_vqa/images/1556507488.jpg",
"ocr_vqa/images/1568811012.jpg",
"ocr_vqa/images/312187114.jpg",
"ocr_vqa/images/435070088.jpg",
"ocr_vqa/images/471243787.jpg",
"ocr_vqa/images/425163636.jpg",
"ocr_vqa/images/789401592.jpg",
"ocr_vqa/images/879802863.jpg",
"ocr_vqa/images/1560445513.jpg",
"ocr_vqa/images/870331612.jpg",
"ocr_vqa/images/1558743030.jpg",
"ocr_vqa/images/937274461.jpg",
"ocr_vqa/images/953735702.jpg",
"ocr_vqa/images/823412385.jpg",
"ocr_vqa/images/966542606.jpg",
"ocr_vqa/images/1883323460.jpg",
"ocr_vqa/images/3797306210.jpg",
"ocr_vqa/images/966586611.jpg",
"ocr_vqa/images/933821131.jpg",
"ocr_vqa/images/1860340660.jpg",
"ocr_vqa/images/668053984.jpg",
"ocr_vqa/images/688121675.jpg",
"ocr_vqa/images/1882419065.jpg",
"ocr_vqa/images/1883323703.jpg",
"ocr_vqa/images/812035844.jpg",
"ocr_vqa/images/185343342X.jpg",
"ocr_vqa/images/1570190380.jpg",
"ocr_vqa/images/879801654.jpg",
"ocr_vqa/images/688118127.jpg",
"ocr_vqa/images/739714864.jpg",
"ocr_vqa/images/1555951295.jpg",
"ocr_vqa/images/739704745.jpg",
"ocr_vqa/images/125476604.jpg",
"ocr_vqa/images/805034676.jpg",
"ocr_vqa/images/761307842.jpg",
"ocr_vqa/images/156458321X.jpg",
"ocr_vqa/images/29344506.jpg",
"ocr_vqa/images/782118577.jpg",
"ocr_vqa/images/3575228701.jpg",
"ocr_vqa/images/764504665.jpg",
"ocr_vqa/images/60191341.jpg",
"ocr_vqa/images/879514795.jpg",
"ocr_vqa/images/785807209.jpg",
"ocr_vqa/images/1567615317.jpg",
"ocr_vqa/images/789401509.jpg",
"ocr_vqa/images/1563053128.jpg",
"ocr_vqa/images/929712838.jpg",
"ocr_vqa/images/717281434.jpg",
"ocr_vqa/images/201624508.jpg",
"ocr_vqa/images/B00XLZW19O.jpg",
"ocr_vqa/images/316051772.jpg",
"ocr_vqa/images/930016238.jpg",
"ocr_vqa/images/051770353X.jpg",
"ocr_vqa/images/B00XLX3W9O.jpg",
"ocr_vqa/images/B01577TUTC.jpg",
"ocr_vqa/images/849930987.jpg",
"ocr_vqa/images/1574770225.jpg",
"ocr_vqa/images/771573936.jpg",
"ocr_vqa/images/316142778.jpg",
"ocr_vqa/images/1560523573.jpg",
"ocr_vqa/images/930410629.jpg",
"ocr_vqa/images/051763547X.jpg",
"ocr_vqa/images/938076140.jpg",
"ocr_vqa/images/1570761116.jpg",
"ocr_vqa/images/962600350.jpg",
"ocr_vqa/images/394532643.jpg",
"ocr_vqa/images/1561700940.jpg",
"ocr_vqa/images/135707978.jpg",
"ocr_vqa/images/570035651.jpg",
"ocr_vqa/images/966424603.jpg",
"ocr_vqa/images/819183482.jpg",
"ocr_vqa/images/963479903.jpg",
"ocr_vqa/images/811726819.jpg",
"ocr_vqa/images/037550267X.jpg",
"ocr_vqa/images/002862405X.jpg",
"ocr_vqa/images/525941290.jpg",
"ocr_vqa/images/664220789.jpg",
"ocr_vqa/images/70224889.jpg",
"ocr_vqa/images/679761799.jpg",
"ocr_vqa/images/393701719.jpg",
"ocr_vqa/images/1560984503.jpg",
"ocr_vqa/images/945397690.jpg",
"ocr_vqa/images/785270965.jpg",
"ocr_vqa/images/762704519.jpg",
"ocr_vqa/images/933478186.jpg",
"ocr_vqa/images/1564583031.jpg",
"ocr_vqa/images/521256771.jpg",
"ocr_vqa/images/590366742.jpg",
"ocr_vqa/images/749521643.jpg",
"ocr_vqa/images/739715593.jpg",
"ocr_vqa/images/899330266.jpg",
"ocr_vqa/images/60170921.jpg",
"ocr_vqa/images/299130002.jpg",
"ocr_vqa/images/1566866391.jpg",
"ocr_vqa/images/894550225.jpg",
"ocr_vqa/images/1570282358.jpg",
"ocr_vqa/images/898796008.jpg",
"ocr_vqa/images/810980789.jpg",
"ocr_vqa/images/1564582914.jpg",
"ocr_vqa/images/823929493.jpg",
"ocr_vqa/images/1890916196.jpg",
"ocr_vqa/images/962289027.jpg",
"ocr_vqa/images/785805516.jpg",
"ocr_vqa/images/188673206X.jpg",
"ocr_vqa/images/1879505460.jpg",
"ocr_vqa/images/883880075.jpg",
"ocr_vqa/images/877936293.jpg",
"ocr_vqa/images/785808841.jpg",
"ocr_vqa/images/1570280762.jpg",
"ocr_vqa/images/465069347.jpg",
"ocr_vqa/images/1568590644.jpg",
"ocr_vqa/images/780800370.jpg",
"ocr_vqa/images/871563932.jpg",
"ocr_vqa/images/800756614.jpg",
"ocr_vqa/images/942627458.jpg",
"ocr_vqa/images/1557488789.jpg",
"ocr_vqa/images/143118153.jpg",
"ocr_vqa/images/1285870298.jpg",
"ocr_vqa/images/1852847190.jpg",
"ocr_vqa/images/130510191X.jpg"]


class LlavaQwDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, image_path: str,
                 data_path:str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 ):
        super(LlavaQwDataset, self).__init__()
        self.image_path = image_path
        list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer

        list_data_dict = [item for item in list_data_dict if 'image' in item and item['image'] not in excluded_image]

        self.list_data_dict = list_data_dict

        # for item in list_data_dict:
        #     length = 0
        #     conversation_str = ''
        #     for conversation in item['conversations']:
        #         length += len(conversation['value'])
        #         conversation_str += conversation['value']
            # if length < 3000:
            #     self.list_data_dict.append(item)

        self.START_IDS = self.tokenizer('<|im_start|>assistant\n')['input_ids']
        self.END_IDS = self.tokenizer.convert_tokens_to_ids('<|im_end|>')

    def __len__(self):
        return len(self.list_data_dict)

    def token_data(self, data):
        image_list = data['data_image_lists']
        converstaion = data['conversations']
        inputs_with_mask = self.tokenizer(converstaion)
        inputs = inputs_with_mask['input_ids']
        labels = [-100] * len(inputs)
        
        for i in range(len(labels)):
            if inputs[i - len(self.START_IDS): i] == self.START_IDS:
                j = inputs.index(self.END_IDS, i)
                for k in range(i, j+1):
                    labels[k] = inputs[k]

        return dict(
            pixel_values = image_list,
            input_ids = inputs,
            attention_mask = inputs_with_mask['attention_mask'],
            labels=labels,
        )


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        source = self.list_data_dict[i]

        data={}
        data_image_lists = []

        template = get_conv_template("chatml")


        if 'image' in source:
            image_file = self.list_data_dict[i]['image']
            data_image_lists.append(os.path.join(self.image_path,image_file))


        conversations = source['conversations']

        for conversation in conversations:
            if conversation['from'] == 'human':
                conversation['value'] = conversation['value'].replace('\n','')
                if '<image>' in conversation['value']:
                    conversation['value'] = conversation['value'].replace('<image>',IMG_START_TOKEN + "<tile_global_thumbnail>" + IMG_CONTEXT_TOKEN * 256 + IMG_END_TOKEN)
                
                template.append_message(template.roles[0], conversation['value'])
            else:
                template.append_message(template.roles[1], conversation['value'])
        
        query = template.get_prompt()
    

        data["data_image_lists"] = data_image_lists
        data["conversations"] = query


        return self.token_data(data)
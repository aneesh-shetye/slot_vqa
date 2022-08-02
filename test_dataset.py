import cv2

from datasets.test_phrasecut import build

from transformers import CLIPFeatureExtractor
from torchvision import transforms as T

clip_feat = CLIPFeatureExtractor(do_resize=False, do_center_crop=False)
clip_feat_extractor = lambda img: clip_feat(img)['pixel_values'][0]
dataset = build(root='/home/aneesh/datasets/PhraseCutDataset/data/VGPhraseCut_v0', 
    resolution = [400, 400], transform= [clip_feat_extractor ])

# for n, i in enumerate(dataset): 
#     print(i['img'].shape)
#     print(i['mask'].shape)
#     trg_obj = i['img']*i['mask']
#     cv2.imwrite('test_mask.jpg', trg_obj.permute(1,2,0).numpy())
#     print(i['phrase'])
#     break

item = dataset[0]
trg_obj = item['img']*item['mask']
# print(trg_obj.shape)
print(item['phrase'])
cv2.imwrite('test.jpg', trg_obj.permute(1,2,0).numpy())
cv2.imwrite('img.jpg', item['img'].permute(1,2,0).numpy())

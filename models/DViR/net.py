import torch
import torch.nn as nn
from PIL import Image
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from models.RefCLIP.head import WeakREChead
from models.network_blocks import MultiScaleFusion
import open_clip
from torch.nn import ModuleList, Linear
import torch.nn.functional as F
from models.clip_encoder import CLIPVisionTower
from transformers import logging
from transformers import Dinov2Model
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from EfficientSAM.efficient_sam.efficient_sam import EfficientSam
logging.set_verbosity_error()

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)

        self.linear_vs = nn.Linear(1024, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)      
        self.head = WeakREChead(__C)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        self.linear = nn.Linear(768,1024)
        self.linear_sam= nn.Linear(256,1024)
        self.clip_model = CLIPVisionTower(vision_tower = "clip model path",input_image_size=224, select_layer=12)
        self.clip_model.eval()
        self.dino_model = Dinov2Model.from_pretrained('dinov2 model path')
        self.dino_model.eval()
        self.efficientsam = build_efficient_sam_vitt()
        self.efficientsam.eval()
        self.convnext_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('clip_convnext model path')
        self.convnext_model.eval()
        self.linear_router = nn.Linear(1024,4)
        self.lambda_sparse = 0.01
        self.linear_decoder = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)  
        )
        self.projector_expert = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )
        self.projector_yolo = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )

        self.class_num = __C.CLASS_NUM
        if __C.VIS_FREEZE:
            self.frozen(self.efficientsam)        
        if __C.VIS_FREEZE:
            self.frozen(self.convnext_model)
        if __C.VIS_FREEZE:
            self.frozen(self.clip_model)
        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)
        if __C.VIS_FREEZE:
            self.frozen(self.dino_model)
    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def re_normalize(self, x):

        # original normalization parameters
        original_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        original_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        # new normalization parameters
        new_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        new_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)

        # unnormalize
        x_unnormalized = x * original_std + original_mean

        # use new normalization parameters
        x_new_pro = (x_unnormalized - new_mean) / new_std
    
        return x_new_pro
    
    def forward(self, x, y):

        # Vision and Language Encoding
        with torch.no_grad():
            boxes_all, x_, boxes_sml = self.visual_encoder(x)
        y_ = self.lang_encoder(y)
       
        x_new_pro = self.re_normalize(x)
        
        resized_image_feature_dino = F.interpolate(x, size=(364, 364), mode='bilinear', align_corners=False)
        resized_image_feature_sam = F.interpolate(x, size=(416, 416), mode='bilinear', align_corners=False)     
          
        # load clip model
        with torch.no_grad():
            clip_feature = self.clip_model(x_new_pro).to(x.device)
        clip_feature = self.linear(clip_feature)
        clip_feature = clip_feature.permute(0, 2, 1).view(clip_feature.size(0), clip_feature.size(2), 13, 13)
        
        #load dino model
        with torch.no_grad():
            dino_feature = self.dino_model(resized_image_feature_dino).last_hidden_state.to(x.device)
        dino_feature = dino_feature[:,1:,:]
        dino_feature = dino_feature.transpose(1, 2).contiguous() .view(dino_feature.size(0),dino_feature.size(2),26,26)
        dino_feature = F.avg_pool2d(dino_feature, kernel_size=2, stride=2)  
        dino_feature = self.linear(dino_feature.permute(0,2,3,1)).permute(0,3,1,2)
        
        #load convnext model
        with torch.no_grad():
            convnext_feature = self.convnext_model.encode_image(x_new_pro).to(x.device)

        convnext_feature = self.linear(convnext_feature.permute(0,2,3,1)).permute(0,3,1,2)
        #load sam model
        with torch.no_grad():  # 
            sam_feature = self.efficientsam.image_encoder(resized_image_feature_sam).to(x.device)

        sam_feature = self.linear_sam(sam_feature.permute(0,2,3,1))
        sam_feature = sam_feature.permute(0,3,1,2)
        sam_feature = F.avg_pool2d(sam_feature, kernel_size=2, stride=2)

        
        # Vision Multi Scale Fusion
        s, m, l = x_
        x_input = [l, m, s]
        l_new, m_new, s_new = self.multi_scale_manner(x_input)
        
        s_new_original = s_new.clone()
        
        dynamic_features = [clip_feature, dino_feature, sam_feature, convnext_feature]
        
       

        yolo_feature = F.adaptive_avg_pool2d(s_new, (1, 1)).permute(0,2,3,1).squeeze(1) #(64, 1,1024)


        # Calculate the probability distribution of router_logits
        router_logits = self.linear_router(yolo_feature.detach()).squeeze(1)
        router_logits= torch.softmax(router_logits,dim=-1)
        

        top1_probs, top1_idx = torch.topk(router_logits, 2, dim=1) 

        selected_features = torch.stack(dynamic_features, dim=1)
        selected_feature1 = selected_features[torch.arange(selected_features.size(0)), top1_idx[:, 0]]


        # Get the value of router_logits for the corresponding index
        batch_size = router_logits.size(0)
        selected_logits0 = router_logits[torch.arange(batch_size), top1_idx[:, 0]]
        selected_logits1 = router_logits[torch.arange(batch_size), top1_idx[:, 1]] 



        logits_sum = selected_logits0 + selected_logits1 
        selected_logits0 = selected_logits0 / logits_sum
        selected_logits1 = selected_logits1 / logits_sum


        s_new = s_new * selected_logits0[:, None, None, None] + \
                selected_feature1 * selected_logits1[:, None, None, None]            

        z_expert = self.projector_expert(selected_feature1)
        z_yolo = self.projector_yolo(s_new_original)

        temperature = 0.5
        batch_size = z_expert.size(0)
        z_expert_norm = F.normalize(z_expert, dim=1)
        z_yolo_norm = F.normalize(z_yolo, dim=1)
        similarity_matrix = torch.matmul(z_expert_norm, z_yolo_norm.T)
        positive_sample = torch.diag(similarity_matrix)
        labels = torch.arange(batch_size).to(x.device)
        criterion = nn.CrossEntropyLoss()
        logits = similarity_matrix / temperature
        loss_contrastive = criterion(logits, labels)
        x_ = [s_new, m_new, l_new]

        # Anchor Selection
        boxes_sml_new = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
        mean_i = mean_i.squeeze(2)[:, :, 4]
        vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, gridnum, anncornum, ch = boxes_sml[0].shape
        bs_, selnum = indices.shape
        box_sml_new = boxes_sml[0].masked_select(
            torch.zeros(bs, gridnum).to(boxes_sml[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                3).expand(bs, gridnum, anncornum, ch)).contiguous().view(bs, selnum, anncornum, ch)
        boxes_sml_new.append(box_sml_new)

        batchsize, dim, h, w = x_[0].size()
        i_new = x_[0].view(batchsize, dim, h * w).permute(0, 2, 1)
        bs, gridnum, ch = i_new.shape
        i_new = i_new.masked_select(
            torch.zeros(bs, gridnum).to(i_new.device).scatter(1, indices, 1).
                bool().unsqueeze(2).expand(bs, gridnum,ch)).contiguous().view(bs, selnum, ch)
        
        recon_text_feature = self.linear_decoder(i_new)
        #pooling
        recon_text_feature_pooled = recon_text_feature.mean(dim=1,keepdim=True)
        recon_loss = F.mse_loss(recon_text_feature_pooled, y_['flat_lang_feat'].unsqueeze(1))
        
        # Anchor-based Contrastive Learning
        x_new = self.linear_vs(i_new)
        y_new = self.linear_ts(y_['flat_lang_feat'].unsqueeze(1))
        if self.training:
            loss = self.head(x_new, y_new)
            total_loss = loss + recon_loss + self.lambda_sparse * sparse_loss + loss_contrastive
            return total_loss
        else:
            predictions_s = self.head(x_new, y_new)
            predictions_list = [predictions_s]
            box_pred = get_boxes(boxes_sml_new, predictions_list,self.class_num)
            return box_pred

def get_boxes(boxes_sml, predictionslist,class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    for i in range(len(predictionslist)):
        mask = predictionslist[i].squeeze(1)
        masked_pred = boxes_sml[i][mask]
        refined_pred = masked_pred.view(batchsize, -1, class_num+5)
        refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
        refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
        refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
        refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
        pred.append(refined_pred.data)
    boxes = torch.cat(pred, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    box_new = torch.gather(boxes, 1, ind_new)
    return box_new
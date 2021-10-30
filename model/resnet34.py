!pip install /kaggle/input/segmentation-models-wheels/efficientnet_pytorch-0.6.3-py3-none-any.whl
!pip install /kaggle/input/segmentation-models-wheels/pretrainedmodels-0.7.4-py3-none-any.whl
!pip install /kaggle/input/segmentation-models-wheels/timm-0.3.2-py3-none-any.whl
!pip install /kaggle/input/segmentation-models-wheels/segmentation_models_pytorch-0.1.3-py3-none-any.whl

!mkdir -p /root/.cache/torch/hub/checkpoints/
!cp ../input/pytorch-pretrained-image-models/resnet34.pth /root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth

import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
import segmentation_models_pytorch as smp 

def build_model(model_name="resnet34"):
   model = smp.Unet(model_name, encoder_weights="imagenet", activation=None)
   return model 
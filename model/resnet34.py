import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
import segmentation_models_pytorch as smp 

def build_model(model_name="resnet34"):
   model = smp.Unet(model_name, encoder_weights="imagenet", activation=None)
   return model 
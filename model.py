import segmentation_models_pytorch as smp

def create_model(num_classes):
    model = smp.FPN(
        encoder_name="timm-mobilenetv3_large_minimal_100",       
        encoder_weights="imagenet",        
        in_channels=3,                     
        classes=num_classes                
    )
    return model

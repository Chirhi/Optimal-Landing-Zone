import segmentation_models_pytorch as smp

def create_model(num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",       
        encoder_weights="imagenet",        
        in_channels=3,                     
        classes=num_classes                
    )
    return model

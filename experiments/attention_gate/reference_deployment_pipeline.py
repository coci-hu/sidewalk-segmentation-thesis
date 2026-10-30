"""Conservative deployment path retaining the frozen torchvision and NumPy operations."""
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

HEIGHT,WIDTH=768,1024
TRANSFORM=transforms.Compose([transforms.Resize((HEIGHT,WIDTH)),transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def read_bgr(path):
    with Image.open(path) as source: rgb=np.array(source.convert('RGB'))
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

def preprocess_bgr(frame):
    if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3: raise ValueError('Expected uint8 BGR')
    rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    return TRANSFORM(rgb).unsqueeze(0).contiguous().numpy()

def binary_mask(logits):
    if logits.shape!=(1,2,HEIGHT,WIDTH) or not np.isfinite(logits).all(): raise ValueError('Invalid binary logits')
    return logits.argmax(1)[0].astype(np.uint8)

def restore_mask(mask,frame):
    return cv2.resize(mask,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)

def overlay_bgr(frame,mask):
    if mask.shape!=frame.shape[:2] or mask.dtype!=np.uint8 or not np.isin(mask,[0,1]).all(): raise ValueError('Invalid binary mask')
    overlay=frame.copy(); selected=mask==1
    overlay[selected]=(.6*overlay[selected]+.4*np.array([40,210,40])).astype(np.uint8)
    return overlay

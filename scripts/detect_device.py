import torch    

def detect_processor():
    if torch.cuda.is_available():
        current_device_index = torch.cuda.current_device()
        gpu_name =  torch.cuda.get_device_name(current_device_index)
        print(f"DEVICE, {gpu_name} detected")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('DEVICE,MPS backend detected')
    else: 
       print("DEVICE,GPU not found" )
try:
    detect_processor()
except:
    print("DEVICE,Failed to get device info")
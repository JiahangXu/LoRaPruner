import time
import torch
print(torch.cuda.device_count())
for i in range(60*24*14):
    time.sleep(60)
    print(i)

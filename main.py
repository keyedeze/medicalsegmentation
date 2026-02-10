import os
from totalsegmentator.python_api import totalsegmentator
#import torch 
#print('CUDA available:', torch.cuda.is_available())
if __name__ == "__main__":
    input_file = "example_mr_sm.nii.gz"
    output_file = "segmentations_mr_sm"

    totalsegmentator(
        input=input_file,
        output=output_file,
        fast=False, 
        preview = False,
        task="total", 
        ml=True, 
        verbose=True
    )

    print(f"Solutions: {output_file}")
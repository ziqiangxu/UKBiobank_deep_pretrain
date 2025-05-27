
import argparse
import os
import sys
import time

import ants
import torch
import numpy as np
from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu


def register_to_mni(image_path):
    """
    Register input image to MNI152 1mm space using ANTs
    Args:
        image_path: Path to input nifti file
    Returns:
        Registered image data as numpy array
    """
    # Load fixed and moving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixed_path = os.path.join(script_dir, 'templates/MNI152_T1_1mm.nii.gz')
    if not os.path.exists(fixed_path):
        fixed_path = ants.get_ants_data('mni')
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(image_path)
    
    # Run registration
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyN',
        # reg_iterations=[100,50,30]
    )
    
    # Get registered image data
    warped_img = reg['warpedmovout']
    return warped_img.numpy()


def predict_brain_age(data, true_age, model_path='./brain_age/run_20190719_00_epoch_best_mae.p'):
    """
    Predict brain age from a registered MRI image
    Args:
        nifti_path: Path to input nifti file
        true_age: True age of the subject
        model_path: Path to the trained model weights
    Returns:
        predicted_age: Predicted brain age
        loss: KL divergence loss between prediction and true age
    """
    # Load model
    model = SFCN()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    
    # Normalize and crop
    data = data/data.mean()
    data = dpu.crop_center(data, (160, 192, 160))
    
    # Prepare input tensor
    sp = (1,1) + data.shape
    data = data.reshape(sp)
    input_data = torch.tensor(data, dtype=torch.float32).cuda()

    # Prepare label
    bin_range = [true_age - 20, true_age + 20]
    bin_step = 1
    sigma = 1
    y, bc = dpu.num2vect(np.array([true_age]), bin_range, bin_step, sigma)
    y = torch.tensor(y, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output = model(input_data)

    # Process output
    x = output[0].cpu().reshape([1, -1])
    loss = dpl.my_KLDivLoss(x, y).numpy()

    x = x.numpy().reshape(-1)
    prob = np.exp(x)
    pred_age = prob@bc

    return pred_age, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict brain age from MRI image')
    parser.add_argument('nifti_path', type=str, help='Path to input nifti file')
    parser.add_argument('true_age', type=float, help='True age of the subject')
    parser.add_argument('--output', '-o', type=str, help='Path to output JSON file')
    
    args = parser.parse_args()
    
    start_time = time.time()
    data = register_to_mni(args.nifti_path)
    reg_time = time.time() - start_time
    print(f'Registration time: {reg_time:.2f} seconds')

    start_time = time.time()
    pred_age, loss = predict_brain_age(data, args.true_age)
    pred_time = time.time() - start_time
    print(f'Prediction time: {pred_time:.2f} seconds')

    print(f'True age: {args.true_age:.1f}')
    print(f'Predicted age: {pred_age:.1f}')
    print(f'KL divergence loss: {loss:.4f}')

    if args.output:
        import json
        results = {
            'nifti_path': args.nifti_path,
            'true_age': args.true_age,
            'predicted_age': float(pred_age),
            'kl_loss': float(loss),
            'registration_time': reg_time,
            'prediction_time': pred_time
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
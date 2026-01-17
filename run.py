import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from rgb_anything_v1.dpt import RGBAnything


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGBAnything')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits'], help='vits available only')
    
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--channel-only', dest='channel_only', action='store_true', help='only display the predicted channel')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='when channel-only, display channel in grayscale')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = RGBAnything(**model_configs[args.encoder])
    model.load_state_dict(torch.load(f'checkpoints/{args.encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_img = cv2.imread(filename)
        
        img = model.infer_image(raw_img, args.input_size)
        
        if args.channel_only:
            if args.grayscale:
                img_g = np.repeat(img[..., np.newaxis], 3, axis=-1)
                raw_g = np.repeat(raw_img[:, :, 1, np.newaxis], 3, axis=-1)

            else:
                img_g = np.stack([np.zeros_like(img), img, np.zeros_like(img)], axis=-1)
                raw_g = np.stack([np.zeros_like(img), raw_img[:, :, 1], np.zeros_like(img)], axis=-1)
            split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_g, split_region, img_g])
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
        else:
            img = np.stack([raw_img[:, :, 0], img, raw_img[:, :, 2]], axis=-1)
            split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_img, split_region, img])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
        
        # if args.pred_only:
            # cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        # else:
            # split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            # combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            # cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
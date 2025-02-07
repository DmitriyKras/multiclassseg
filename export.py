import argparse
import os
import torch
from onnxsim import simplify
import onnx
# import models
from models import EfficientUnet, ESNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', 
                        help="""Segmentation model name. Available models: effunet, esnet""")
    parser.add_argument('--n-classes', type=int, default=1, 
                        help="""Number of classes in dataset""")
    parser.add_argument('--img', type=int, default=512, 
                        help='export image size (pixels)')
    parser.add_argument('--weights', type=str, default='weights/effunet/best.pt',
                        help='path to weight file to validate')
    parser.add_argument('--onnx-file', type=str, default='onnx/effunet/model.onnx',
                        help='path of exported onnx file')
    parser.add_argument('--opset', type=int, default=11,
                        help='desired onnx opset version')
    args = parser.parse_args()

    # create folders if not exist
    if not os.path.exists('onnx'):
        os.makedirs('onnx')
    if not os.path.exists(f'onnx/{args.model}'):
        os.makedirs(f'onnx/{args.model}')
    
    ### GET MODEL ###

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
    if torch.cuda.is_available():
        print('[INFO] Using CUDA device')
    else:
        print('[INFO] CUDA is not detected. Using CPU')
    print('[INFO] Loading model...')
    if args.model == 'effunet':
        model = EfficientUnet(args.n_classes).to(device)
    elif args.model == 'esnet':
        model = ESNet(args.n_classes).to(device)
    else:
        print(f'[INFO] Model {args.model} is not supported. Using default effunet')
        model = EfficientUnet(args.n_classes).to(device)
    model.load_state_dict(torch.load(args.weights))
    print('[INFO] Model successfully loaded. Exporting to onnx...')
    model.eval()
    # trace model
    x = torch.rand(1, 3, args.img, args.img, device=device, requires_grad=True)
    y = model(x)
    # export
    torch.onnx.export(
    model,
    x,
    args.onnx_file,
    opset_version=args.opset,
    input_names=['image'],
    output_names=['mask'],
    do_constant_folding=True
    )
    print('[INFO] Model successfully exported. simplifying onnx file...')
    # simplify model
    onnx_model = onnx.load(args.onnx_file)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.onnx_file)
    print('[INFO] Final onnx file created')

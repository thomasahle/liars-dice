import torch
import argparse

from snyd import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='+', help="Model to export")
    args = parser.parse_args()

    for path in args.path:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        cpargs = checkpoint["args"]

        # Model : (private state, public state) -> value
        D_PUB, D_PRI, *_ = calc_args(cpargs.d1, cpargs.d2, cpargs.sides, cpargs.variant)
        #model = Net(D_PRI, D_PUB)
        model = NetCompBilin(D_PRI, D_PUB)
        model.load_state_dict(checkpoint["model_state_dict"])

        dummy_priv = torch.zeros(D_PRI)
        dummy_pub = torch.zeros(D_PUB)
        torch.onnx.export(model, (dummy_priv, dummy_pub), path+'.onnx', verbose=True,
                        input_names=['priv', 'pub'], output_names=['value'])


if __name__ == '__main__':
    main()

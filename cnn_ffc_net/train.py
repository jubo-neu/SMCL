import matplotlib.pyplot as plt
import argparse

import numpy as np
import torch
import tqdm

from data import get_data
from networks import get_model, FFCNet, FFCNett
from losses import InfoNCELoss, FocalFrequencyLoss, FocalFrequencyLoss2

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_iterations', default=200, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--ffc_lambda', default=0, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--dataset', default='lamp', choices=["bench", "car", "chair", "display", "plane", "speaker", "lamp"])
    parser.add_argument('--image_size', default='64', type=int)
    parser.add_argument('--image_dir', default="C:/Users/Lenovo/Desktop/Mir-180_Dataset/lamp")
    parser.add_argument('--model_name', default="cnn_ffc", choices=["cnn_cnn", "cnn_ffc"])
    parser.add_argument('--g_ratio', default=0.5, type=float)
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)

    return parser


def colored_text(st):
    return '\033[91m' + st + '\033[0m'


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def eval(val_loader, criteria, model, FFCNet, FFCNett, device="cuda"):
    model.eval()
    FFCNet.eval()
    FFCNett.eval()
    total_loss = 0
    losses = {criterion.__class__.__name__: 0 for criterion in criteria}
    counter = 0

    with torch.no_grad():
        for img, label_img, neg_label_img, _, _, _ in tqdm.tqdm(val_loader):

            img_m = model(img.to(device))
            label_m = model(label_img.to(device))
            neg_label_m = model(neg_label_img.to(device))

            img_f = FFCNet(img.to(device))
            label_f = FFCNet(label_img.to(device))

            img_ff = FFCNett(img.to(device))
            neg_label_f = FFCNett(neg_label_img.to(device))

            for criterion in criteria:
                if isinstance(criterion, InfoNCELoss):
                    loss = criterion(img_m, label_m, neg_label_m)
                elif isinstance(criterion, FocalFrequencyLoss):
                    loss = criterion(img_f, label_f)
                elif isinstance(criterion, FocalFrequencyLoss2):
                    loss = criterion(img_ff, neg_label_f)
                else:
                    raise NotImplementedError("Unsupported criterion type")

                losses[criterion.__class__.__name__] += loss.item()
                total_loss += loss.item()

            counter += 1

    for criterion_name in losses.keys():
        losses[criterion_name] /= counter
        print(f"Validation {criterion_name} loss: {losses[criterion_name]}")

    total_loss /= counter
    print("Total Validation loss: ", total_loss)
    return total_loss


def train(args):
    device = args.device
    model_name = args.model_name
    dataset_name = args.dataset
    learning_rate = args.learning_rate
    ratio = args.g_ratio
    data_path = args.image_dir
    iterations = args.num_iterations
    img_size = args.image_size
    batch_size = args.batch_size

    criterion_neg = InfoNCELoss()
    criterion_ffc = FocalFrequencyLoss()
    criterion_ffc2 = FocalFrequencyLoss2()

    save_name = model_name + "_" + dataset_name + ".pt"

    model = get_model(model_name, ratio=ratio).to(device)
    model.train()
    FFCNet1 = FFCNet().to(device)
    FFCNet1.train()
    FFCNet2 = FFCNett().to(device)
    FFCNet2.train()

    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=args.weight_decay)
    optimizer_FFCNet1 = torch.optim.Adam(list(FFCNet1.parameters()), lr=0.0001, weight_decay=args.weight_decay)
    optimizer_FFCNet2 = torch.optim.Adam(list(FFCNet2.parameters()), lr=0.0001, weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader, _, _, _ = get_data(data_path, img_size, batch_size)

    train_losses = [[], [], []]

    for t in range(iterations):
        for img, label_img, neg_label_img, filename, label_filename, neg_label_filename in tqdm.tqdm(train_loader):

            img_m = model(img.to(device))
            label_m = model(label_img.to(device))
            neg_label_m = model(neg_label_img.to(device))

            img_f = FFCNet1(img.to(device))
            label_f = FFCNet1(label_img.to(device))

            img_ff = FFCNet2(img.to(device))
            neg_label_f = FFCNet2(neg_label_img.to(device))

            Loss_neg = criterion_neg(img_m, label_m, neg_label_m)

            optimizer.zero_grad()
            Loss_neg.backward()
            optimizer.step()

            Loss_ffc = criterion_ffc(img_f, label_f)

            optimizer_FFCNet1.zero_grad()
            Loss_ffc.backward()
            optimizer_FFCNet1.step()

            Loss_ffc2 = criterion_ffc2(img_ff, neg_label_f)

            optimizer_FFCNet2.zero_grad()
            Loss_ffc2.backward()
            optimizer_FFCNet2.step()


        if t % 1 == 0:
            print("InfoNCELoss loss for model: ", Loss_neg.item())
            print("FocalFrequencyLoss loss for FFCNet1 positive:", Loss_ffc.item())
            print("FocalFrequencyLoss loss for FFCNet2 negative:", Loss_ffc2.item())
            print("Epoch", t, "/", iterations)

            train_losses[0].append(Loss_neg.item())
            train_losses[1].append(Loss_ffc.item())
            train_losses[2].append(Loss_ffc2.item())

            criterion_neg = InfoNCELoss()
            criterion_ffc = FocalFrequencyLoss()
            criterion_ffc2 = FocalFrequencyLoss2()

            criteria = [criterion_neg, criterion_ffc, criterion_ffc2]

        if t % 10 == 0:
            print("Validation")
            validation_loss = eval(val_loader, criteria, model, FFCNet1, FFCNet2)

    torch.save(model.state_dict(), save_name)

    save_path_neg = "your path"
    save_path_ffc = "your path"
    save_path_ffc2 = "your path"

    plt.figure()
    plt.plot(train_losses[0], label='Loss_neg')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('InfoNCE Loss')
    plt.legend()
    plt.savefig(save_path_neg)

    plt.figure()
    plt.plot(train_losses[1], label='Loss_ffc')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('FocalFrequency Loss for FFCNet1')
    plt.legend()
    plt.savefig(save_path_ffc)

    plt.figure()
    plt.plot(train_losses[2], label='Loss_ffc2')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('FocalFrequency Loss for FFCNet2')
    plt.legend()
    plt.savefig(save_path_ffc2)

    return model, validation_loss


if __name__ == "__main__":
    args = argument_parser().parse_args()
    print(args)
    set_seed(args.seed)

    train(args)

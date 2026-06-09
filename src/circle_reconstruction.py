import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


#filename circles_numberOfCircles_ImageSize_noiseType_maxNumberOFCirclesInImage_MeanSizeMeanShiftXMeanShiftY
#FILENAME = "circles_30k_64_wn_sc_106464"
FILENAME = "circles_10_64_wn_sc_106464"

circle_creation = True
numberImages = 10

max_number_circles = 3
multiple_circles = True

#network either "Linear", "UNET" or "GAN"
network = "Linear"

LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 1
IM_SIZE = 64

#--------------------------------------------------------------------------------------------------------------

#make circle map defined by radius and offsets
def make_circle_map(map_size: int = 32, radius: float = 10, cx: float = 16, cy: float = 16):
    x = np.linspace(0, map_size-1, map_size)
    y = np.linspace(0, map_size-1, map_size)
    X, Y = np.meshgrid(x, y)
    Z = (((X-cx)**2 + (Y-cy)**2) <= radius**2).astype(int)
    return Z

#make multiple (single) circle maps with random radii and offsets + documenting circle on each map
def make_random_circle_maps(n: int = 1, map_size: int = 64, radius_mean: float = 10, radius_var: float = 5, radius_min: float = 2, 
                            cx_mean: float = 32, cx_var: float = 20, cy_mean: float = 32, cy_var: float = 20, max_shift: float = 25, seed: int = 42):
    rng = np.random.default_rng(seed)
    radii = rng.normal(radius_mean, radius_var, n).clip(min = radius_min)
    cx_shifts = rng.normal(cx_mean, cx_var, n).clip(max=max_shift)
    cy_shifts = rng.normal(cy_mean, cy_var, n).clip(max=max_shift)

    circle_images = np.zeros((n, map_size, map_size))
    doc = []
    for i in range(n):
        circle_images[i, :, :] = make_circle_map(map_size, radii[i], cx_shifts[i], cy_shifts[i])
        doc.append({"index": i, "radius": round(radii[i], 3), "cx_shift": round(cx_shifts[i], 3), "cy_shift": round(cy_shifts[i], 3)})

    return circle_images, doc

#make map with multiple circles, each defined by radius and offset in list
def make_multiple_circle_map(map_size: int = 32, number_circles: int = 1,  radius: list[float] = [10], cx: list[float] = [16], cy: list[float] = [16]):
    x = np.linspace(0, map_size-1, map_size)
    y = np.linspace(0, map_size-1, map_size)
    X, Y = np.meshgrid(x, y)
    Z_list = []
    for i in range(number_circles):
        Z = (((X-cx[i])**2 + (Y-cy[i])**2) <= radius[i]**2).astype(int)
        Z_list.append(Z)
    Z_final = sum(Z_list)
    return Z_final

#make multiple maps with random number of circles given a max number of circles and mean circle parameters 
def make_random_multiple_circle_maps(n: int = 1, map_size: int = 64, max_number_circles: int = 1, radius_mean: float = 10, radius_var: float = 5, radius_min: float = 2, 
                            cx_mean: float = 32, cx_var: float = 20, cy_mean: float = 32, cy_var: float = 20, max_shift: float = 25, seed: int = 42):
    rng = np.random.default_rng(seed)

    circle_images = np.zeros((n, map_size, map_size))
    doc = []

    for i in range(n):
        number_circles = rng.integers(1, max_number_circles, endpoint=True)

        radii = rng.normal(radius_mean, radius_var, number_circles).clip(min=radius_min)
        cx_shifts = rng.normal(cx_mean, cx_var, number_circles).clip(max=max_shift)
        cy_shifts = rng.normal(cy_mean, cy_var, number_circles).clip(max=max_shift)

        circle_images[i, :, :] = make_multiple_circle_map(map_size, number_circles, radii, cx_shifts, cy_shifts)
        doc.append({"index": i, "radius": radii, "cx_shift": cx_shifts, "cy_shift": cy_shifts})

    return circle_images, doc

#make noise white noise map
def make_noise_maps(n: int = 1, map_size: int = 32, noise_max: float = 3, seed: int = 42):
    noise_maps = np.zeros((n, map_size, map_size))
    rng = np.random.default_rng(seed)
    for i in range(n):
        noise_maps[i, :, :] = rng.uniform(0, noise_max, size=(map_size, map_size))

    return noise_maps

#save maps in .npz file
def save_maps(maps, ground_truths, doc, file_name: str = f"../data/circle_maps"):
    
    radii = [m["radius"] for m in doc]
    cx_shifts = [m["cx_shift"] for m in doc]
    cy_shifts = [m["cy_shift"] for m in doc]

    np.savez(file_name + ".npz", maps=maps, ground_truths=ground_truths, radii=np.array(radii, dtype=object), cx_shifts=np.array(cx_shifts, dtype=object), cy_shifts=np.array(cy_shifts, dtype=object), allow_pickle=True)
    print("File saved as ", file_name, ".npz")
    return None

#--------------------------------------------------------------------------------------------------------------

#Load .npz files and return Noise+CircleMaps, CircleMaps, parameters
def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    print(f"Keys in {path}: {list(data.keys())}")
 
    X = data["maps"]
    Y = data["ground_truths"]
    y = np.column_stack([data["radii"], data["cx_shifts"], data["cy_shifts"]])
    
    print(f"Loaded  X: {X.shape}  Y: {Y.shape}  y: {y.shape}")
    return X, Y, y

#--------------------------------------------------------------------------------------------------------------

######              UNET            ######

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels, out_channels,  features=[64, 128, 256, 512], 
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride =2)

        #Down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[int(idx/2)]

            ###TODO: resizing to handle wrong image sizes due to flooring in maxpool layer

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
#--------------------------------------------------------------------------------------------------------------

######              GAN            ######
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNET_generator(nn.Module):
    def __init__(
            self, in_channels, out_channels,  features=[64, 128, 256, 512], 
    ):
        super(UNET_generator, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride =2)

        #Down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[int(idx/2)]

            ###TODO: resizing to handle wrong image sizes due to flooring in maxpool layer

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

#--------------------------------------------------------------------------------------------------------------
######              SimpleNet            ######

class SimpleNet(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        total_loss += loss_fn(model(X_batch), y_batch).item() * len(X_batch)
    return total_loss / len(loader.dataset)
    
#--------------------------------------------------------------------------------------------------------------

def main():
    if circle_creation:
        if multiple_circles:
            circle_maps, doc = make_random_multiple_circle_maps(numberImages, IM_SIZE, max_number_circles)
        else:
            circle_maps, doc = make_random_circle_maps(numberImages, IM_SIZE)
        noise_maps = make_noise_maps(numberImages, IM_SIZE)
        maps = circle_maps+noise_maps
        save_maps(maps, circle_maps, doc, f"../data/{FILENAME}")

    data, sol_maps, sol = load_npz(f"../data/{FILENAME}.npz")
    if network == "UNET":

        #loading and preparing data
        data_tensor = torch.from_numpy(data).float().unsqueeze(1)
        sol_tensor = torch.from_numpy(sol_maps).float().unsqueeze(1)
        dataset = TensorDataset(data_tensor, sol_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        #training model
        model = UNET(in_channels=1, out_channels=1).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()  # or nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        print(f"Training on {DEVICE}  |  {IM_SIZE}×{IM_SIZE} maps  |  batch={BATCH_SIZE}  |  epochs={NUM_EPOCHS}    |   learningrate={LR}\n")
        print("-" * 32)
        model.train()
        for epoch in range(NUM_EPOCHS):
            running_loss = 0

            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                #forward
                preds = model(x)
                loss = criterion(preds, y)

                #backwards
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss/len(dataloader)
            print("Epoch: ", epoch, " Loss: ", avg_loss)

        print("Done!")

        #evaluation
        #TODO: compute eval metrics 
        model.eval()

        with torch.no_grad():
            x, y = next(iter(dataloader))
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            preds = torch.sigmoid(preds)

        # Plot first image in the batch
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(x[0].squeeze().cpu(), cmap="gray")
        axes[0].set_title("Input")

        axes[1].imshow(preds[0].squeeze().cpu(), cmap="gray")
        axes[1].set_title("Prediction")

        axes[2].imshow(y[0].squeeze().cpu(), cmap="gray")
        axes[2].set_title("Ground Truth")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"../outputs/exampleIMG_{network}_{FILENAME}", dpi=150, bbox_inches="tight")
        plt.close()

    elif network == "GAN":
        disc = Discriminator(IM_SIZE).to(DEVICE)
        gen = UNET_generator(in_channels=1, out_channels=1).to(DEVICE)

        #sol_maps = np.zeros((n, 64, 64))
        #for i in range(n):
        #    sol_maps[i, :, :] = make_circle_map(64, sol[i, 0], sol[i, 1], sol[i, 2])

        data_tensor = torch.from_numpy(data).float().unsqueeze(1)
        sol_tensor = torch.from_numpy(sol_maps).float().unsqueeze(1)
        dataset = TensorDataset(data_tensor, sol_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        opt_disc = optim.Adam(disc.parameters(), lr=LR)
        opt_gen = optim.Adam(gen.parameters(), lr=LR)
        loss = nn.BCEWithLogitsLoss()

        gen.train()
        disc.train()

        for epoch in range(NUM_EPOCHS):
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                #generator
                preds = gen(x)
                loss_gen = loss(preds, y)

                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                #discrimitor
                fake = gen(x)
                disc_real = disc(y).view(-1)
                disc_fake = disc(fake).view(-1)
                lossD_real = loss(disc_real, torch.ones_like(disc_real))
                lossD_fake = loss(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake)/2
                disc.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()

                print("Epoch: " + str(epoch))
                print("Loss D: " + str(lossD))

        gen.eval()  # switch to eval mode
        #TODO: calculate evaluation metrics

        with torch.no_grad():
            x, y = next(iter(dataloader))
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = gen(x)
            preds = torch.sigmoid(preds)

        # Plot first image in the batch
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(x[0].squeeze().cpu(), cmap="gray")
        axes[0].set_title("Input")

        axes[1].imshow(preds[0].squeeze().cpu(), cmap="gray")
        axes[1].set_title("Prediction")

        axes[2].imshow(y[0].squeeze().cpu(), cmap="gray")
        axes[2].set_title("Ground Truth")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"../outputs/exampleIMG_{network}_{FILENAME}", dpi=150, bbox_inches="tight")
        plt.close()

    elif network == "Linear":

        X = torch.tensor(data, dtype=torch.float32)
        # necessary reshaping of parameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
        sol_numeric = np.array([np.concatenate([np.atleast_1d(r) for r in row]) for row in sol], dtype=np.float32)
        y = torch.tensor(sol_numeric, dtype=torch.float32)

        train_X, train_y = X[:28000], y[:28000]
        test_X,  test_y  = X[28000:], y[28000:]

        H, W = X.shape[1], X.shape[2]

        train_loader = DataLoader(TensorDataset(train_X, train_y),
                          batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(TensorDataset(test_X,  test_y),
                                batch_size=BATCH_SIZE, shuffle=False)
        model     = SimpleNet(H, W).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        loss_fn   = nn.MSELoss()

        print(f"Training on {DEVICE}  |  {H}×{W} maps  |  batch={BATCH_SIZE}  |  epochs={NUM_EPOCHS}\n")
        print(f"{'Epoch':>6}  {'Train MSE':>10}  {'LR':>10}")
        print("-" * 32)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train(model, train_loader, optimizer, loss_fn, DEVICE)
            scheduler.step()
            if epoch % 5 == 0 or epoch == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(f"{epoch:>6}  {train_loss:>10.6f}  {lr_now:>10.2e}")

        #test
        test_loss = evaluate(model, test_loader, loss_fn, DEVICE)
        print(f"\nTest MSE  : {test_loss:.6f}")
        print(f"Test RMSE : {test_loss**0.5:.6f}")

        # per-output breakdown
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                all_preds.append(model(X_batch.to(DEVICE)).cpu())
                all_targets.append(y_batch)

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        labels  = ["radius", "cx_shift", "cy_shift"]

        print("\nPer-output RMSE on test set:")
        for i, name in enumerate(labels):
            rmse = ((preds[:, i] - targets[:, i]) ** 2).mean().sqrt().item()
            print(f"  {name:<10}: {rmse:.6f}")
        
        #random forward pass
        idx      = torch.randint(len(test_X), (1,)).item()
        sample_X = test_X[idx].unsqueeze(0).to(DEVICE)
        sample_y = test_y[idx]                             # [radius, cx_shift, cy_shift]

        model.eval()
        with torch.no_grad():
            pred = model(sample_X).squeeze(0).cpu()

        #get parameters
        map_size = H                                       # H == W assumed square

        r_true,  cx_true,  cy_true  = sample_y.tolist()
        r_pred,  cx_pred,  cy_pred  = pred.tolist()

        map_true = make_circle_map(map_size, radius=r_true, cx=cx_true, cy=cy_true)
        map_pred = make_circle_map(map_size, radius=r_pred, cx=cx_pred, cy=cy_pred)

        #make outputs
        labels = ["radius", "cx_shift", "cy_shift"]
        print(f"Sample index: {idx}\n")
        print(f"{'Output':<12} {'Predicted':>12} {'Ground Truth':>14} {'Error':>10}")
        print("-" * 52)
        for name, p, t in zip(labels, pred, sample_y):
            print(f"{name:<12} {p.item():>12.4f} {t.item():>14.4f} {abs(p-t).item():>10.4f}")

        #plot
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        panels = [
            (test_X[idx].numpy(), "viridis", f"Noisy input\n(sample #{idx})"),
            (map_true,            "Blues",   f"Ground truth\nr={r_true:.2f}  cx={cx_true:.2f}  cy={cy_true:.2f}"),
            (map_pred,            "Oranges", f"Predicted\nr={r_pred:.2f}  cx={cx_pred:.2f}  cy={cy_pred:.2f}"),
        ]

        for ax, (data, cmap, title) in zip(axes, panels):
            ax.imshow(data, cmap=cmap, origin="upper")
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.suptitle("Network prediction vs ground truth", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(f"../outputs/exampleIMG_{network}_{FILENAME}", dpi=150, bbox_inches="tight")
    
    return None


if __name__ == "__main__":
    main()
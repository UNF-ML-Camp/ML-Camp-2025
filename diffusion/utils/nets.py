import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m

def conv_block(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
        nn.SiLU()
    )

def down_block(ch):
    return conv_block(ch, ch, stride=2)

def up_block(in_c, out_c):
    return nn.ModuleDict({
        "up":   nn.ConvTranspose2d(in_c, out_c,
                                   kernel_size=2, stride=2,
                                   padding=0, output_padding=0),
        "conv": conv_block(out_c * 2, out_c)
    })

class UNet16(nn.Module):
    """
    UNet that receives:
        x       (B, 3, 768)
        t       (B, 1)
        labels  (B, n_classes)  – one-hot
    and returns:
        (B, 3, 768)
    """
    def __init__(self, n_classes: int, base: int = 64):
        super().__init__()
        self.n_classes = n_classes + 1
        self.img_hw    = 16                # 16 × 16 patches
        img_in_ch      = 3                 # RGB
        in_c_total     = img_in_ch + 1 + n_classes     # image + t + labels

        # ---------------- encoder ----------------
        self.enc0  = conv_block(in_c_total, base)   # 16×16
        self.down0 = down_block(base)               # 8×8

        self.enc1  = conv_block(base, base * 2)     # 8×8
        self.down1 = down_block(base * 2)           # 4×4

        self.enc2  = conv_block(base * 2, base * 4) # 4×4
        self.down2 = down_block(base * 4)           # 2×2

        # ---------------- bottleneck -------------
        self.bottleneck = nn.Sequential(
            conv_block(base * 4, base * 8),          # 2×2
            nn.Conv2d(base * 8, base * 8, kernel_size=1)  # 2×2
        )

        # ---------------- decoder ----------------
        self.up2 = up_block(base * 8, base * 4)     # 2×2 -> 4×4
        self.up1 = up_block(base * 4, base * 2)     # 4×4 -> 8×8
        self.up0 = up_block(base * 2, base)         # 8×8 -> 16×16

        # ---------------- output -----------------
        self.out_conv = nn.Conv2d(base, 3, kernel_size=1)

    # ---------------------------------------------------------------- forward -
    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, conditioning: bool = True):
        """
        x:      (B, 3, 768)           – flattened RGB tile
        t:      (B, 1)                – scalar timestep
        labels: (B, n_classes)        – one-hot row
        """
        B, C, L = x.shape                         # L = 768
        H = W = int(m.sqrt(L))            # -> 16

        # ---------------- reshape & broadcast -----------
        img = x.view(B, C, H, W)                  # (B, 3, 16, 16)
        tt  = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        if conditioning:
            lab = labels.view(B, self.n_classes, 1, 1).expand(B, self.n_classes, H, W)
            tt = tt + lab

        signal = torch.cat([img, tt], dim=1) # (B, C+1, 16, 16)

        # ---------------- encoder ----------------
        e0 = self.enc0(signal)
        d0 = self.down0(e0)

        e1 = self.enc1(d0)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        # ---------------- bottleneck -------------
        b = self.bottleneck(d2)

        # ---------------- decoder ----------------
        u2 = self.up2["up"](b)                       # 4×4
        u2 = self.up2["conv"](torch.cat([u2, e2], 1))

        u1 = self.up1["up"](u2)                      # 8×8
        u1 = self.up1["conv"](torch.cat([u1, e1], 1))

        u0 = self.up0["up"](u1)                      # 16×16
        u0 = self.up0["conv"](torch.cat([u0, e0], 1))

        out = self.out_conv(u0)                      # (B,3,16,16)
        return out.view(B, 3, -1)                    # restore (B,3,768)

def main():
    # Simulate a batch of noisy MNIST-like images
    batch_size = 64
    image_dim = 16
    num_classes = 5
    embed_dim = 3

    # Flattened input vector (e.g. from DDPM)
    # torch.Size([64, 768]) torch.Size([64, 1]) torch.Size([64, 5])
    x = torch.randn(batch_size, embed_dim, image_dim * image_dim)
    print(x.shape)

    # Normalized timestep (e.g. from sinusoidal embedding step)
    t = torch.rand(batch_size, 1)

    # Random integer class labels between 0 and 9
    labels = torch.randint(0, num_classes, (batch_size, num_classes))

    # torch.Size([64, 3, 256])
    # torch.Size([64, 1])
    # torch.Size([64, 5])

    # Initialize the model
    model = UNet16(n_classes=num_classes-1)

    # Forward pass
    with torch.no_grad():
        output = model(x, t, labels)

    # Check the output
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output min/max:", output.min().item(), output.max().item())

if __name__ == "__main__":
    main()
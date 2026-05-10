import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    METRICS_AVAILABLE = True
except ModuleNotFoundError:
    psnr_fn = None
    ssim_fn = None
    METRICS_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = APP_DIR / "ddpm_model.pth"

NATIVE_IMG_SIZE = 32
TIME_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02
BASE_CHANNELS = 64
TIME_DIM = 128


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb_scale = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = BASE_CHANNELS, time_dim: int = TIME_DIM):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)

        self.enc1 = ResBlock(in_ch, base_ch, time_dim * 4)
        self.enc2 = ResBlock(base_ch, base_ch, time_dim * 4)
        self.down1 = Downsample(base_ch)

        self.enc3 = ResBlock(base_ch, base_ch * 2, time_dim * 4)
        self.enc4 = ResBlock(base_ch * 2, base_ch * 2, time_dim * 4)
        self.down2 = Downsample(base_ch * 2)

        self.enc5 = ResBlock(base_ch * 2, base_ch * 4, time_dim * 4)
        self.enc6 = ResBlock(base_ch * 4, base_ch * 4, time_dim * 4)
        self.down3 = Downsample(base_ch * 4)

        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim * 4)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim * 4)

        self.up1 = Upsample(base_ch * 4)
        self.dec1 = ResBlock(base_ch * 8, base_ch * 4, time_dim * 4)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, time_dim * 4)

        self.up2 = Upsample(base_ch * 2)
        self.dec3 = ResBlock(base_ch * 4, base_ch * 2, time_dim * 4)
        self.dec4 = ResBlock(base_ch * 2, base_ch, time_dim * 4)

        self.up3 = Upsample(base_ch)
        self.dec5 = ResBlock(base_ch * 2, base_ch, time_dim * 4)
        self.dec6 = ResBlock(base_ch, base_ch, time_dim * 4)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(e1, t_emb)
        e3 = self.enc3(self.down1(e2), t_emb)
        e4 = self.enc4(e3, t_emb)
        e5 = self.enc5(self.down2(e4), t_emb)
        e6 = self.enc6(e5, t_emb)

        b = self.down3(e6)
        b = self.mid1(b, t_emb)
        b = self.mid2(b, t_emb)

        d = self.up1(b)
        d = self.dec1(torch.cat([d, e6], dim=1), t_emb)
        d = self.dec2(d, t_emb)

        d = self.up2(d)
        d = self.dec3(torch.cat([d, e4], dim=1), t_emb)
        d = self.dec4(d, t_emb)

        d = self.up3(d)
        d = self.dec5(torch.cat([d, e2], dim=1), t_emb)
        d = self.dec6(d, t_emb)

        return self.out_conv(F.silu(self.out_norm(d)))


class DDPMScheduler:
    def __init__(self, timesteps: int = TIME_STEPS, beta_start: float = BETA_START, beta_end: float = BETA_END):
        self.T = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise


def normalize_state_dict(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


@st.cache_resource
def load_model(checkpoint_path: str, device_str: str):
    device = torch.device(device_str)
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(normalize_state_dict(state_dict), strict=True)
    model.eval()
    return model


def to_tensor(image: Image.Image, size: int = NATIVE_IMG_SIZE) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size))
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return ((array + 1.0) / 2.0).clip(0, 1)


def make_step_schedule(total_steps: int, preview_steps: int) -> list[int]:
    values = np.linspace(total_steps - 1, 0, preview_steps)
    steps = sorted({int(round(v)) for v in values}, reverse=True)
    if 0 not in steps:
        steps.append(0)
    return sorted(set(steps), reverse=True)


@torch.no_grad()
def sample_from_noise(model: nn.Module, scheduler: DDPMScheduler, device: torch.device, n_images: int, preview_steps: int, seed: int):
    set_seed(seed)
    x = torch.randn(n_images, 3, NATIVE_IMG_SIZE, NATIVE_IMG_SIZE, device=device)
    intermediates = {}
    preview_set = set(make_step_schedule(scheduler.T, preview_steps))

    for t_val in reversed(range(scheduler.T)):
        t_batch = torch.full((n_images,), t_val, device=device, dtype=torch.long)

        alpha = scheduler.alphas[t_val].to(device)
        alpha_cumprod = scheduler.alpha_cumprod[t_val].to(device)
        alpha_cumprod_prev = scheduler.alpha_cumprod_prev[t_val].to(device)
        beta = scheduler.betas[t_val].to(device)

        noise_pred = model(x, t_batch)
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = beta / torch.sqrt(1.0 - alpha_cumprod)
        x_prev = coef1 * (x - coef2 * noise_pred)

        if t_val > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
            x_prev = x_prev + variance * noise

        x = x_prev
        if t_val in preview_set:
            intermediates[t_val] = x.clone()

    return x, intermediates


@torch.no_grad()
def reconstruct_from_image(model: nn.Module, scheduler: DDPMScheduler, device: torch.device, image: Image.Image, start_t: int, preview_steps: int, seed: int):
    set_seed(seed)
    x0 = to_tensor(image, NATIVE_IMG_SIZE).to(device)
    t = torch.tensor([start_t], device=device, dtype=torch.long)
    x_t, _ = scheduler.add_noise(x0, t)

    x = x_t.clone()
    preview_set = set(make_step_schedule(start_t + 1, preview_steps))
    intermediates = {start_t: x_t.clone()}

    for t_val in reversed(range(start_t + 1)):
        t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)
        alpha = scheduler.alphas[t_val].to(device)
        alpha_cumprod = scheduler.alpha_cumprod[t_val].to(device)
        alpha_cumprod_prev = scheduler.alpha_cumprod_prev[t_val].to(device)
        beta = scheduler.betas[t_val].to(device)

        noise_pred = model(x, t_batch)
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = beta / torch.sqrt(1.0 - alpha_cumprod)
        x_prev = coef1 * (x - coef2 * noise_pred)

        if t_val > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
            x_prev = x_prev + variance * noise

        x = x_prev
        if t_val in preview_set:
            intermediates[t_val] = x.clone()

    return x0, x_t, x, intermediates


def compute_metrics(reference: torch.Tensor, generated: torch.Tensor):
    reference_np = to_image(reference)
    generated_np = to_image(generated)
    if METRICS_AVAILABLE:
        psnr = psnr_fn(reference_np, generated_np, data_range=1.0)
        ssim = ssim_fn(reference_np, generated_np, data_range=1.0, channel_axis=2)
    else:
        mse = float(np.mean((reference_np - generated_np) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
        ssim = None
    return psnr, ssim


def render_step_grid(step_map: dict[int, torch.Tensor], title: str):
    ordered = sorted(step_map.items(), reverse=True)
    fig, axes = plt.subplots(1, len(ordered), figsize=(3 * len(ordered), 3))
    if len(ordered) == 1:
        axes = [axes]

    for axis, (step, tensor) in zip(axes, ordered):
        axis.imshow(to_image(tensor))
        axis.set_title(f"t = {step}")
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def render_image_row(tensors: torch.Tensor, title: str):
    count = tensors.shape[0]
    fig, axes = plt.subplots(1, count, figsize=(3 * count, 3))
    if count == 1:
        axes = [axes]

    for idx, axis in enumerate(axes):
        axis.imshow(to_image(tensors[idx : idx + 1]))
        axis.set_title(f"Image {idx + 1}")
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="DDPM Streamlit App", layout="wide")
    st.title("Diffusion Model Demo")
    st.caption("This app loads the trained DDPM checkpoint from the workspace folder and supports generation, reconstruction, and diffusion visualization.")

    device = get_device()
    scheduler = DDPMScheduler()

    with st.sidebar:
        st.header("Controls")
        checkpoint_path = st.text_input("Checkpoint path", str(DEFAULT_CKPT))
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
        generated_images = st.slider("Generated images", min_value=1, max_value=5, value=5)
        preview_steps = st.slider("Preview steps", min_value=3, max_value=7, value=5)
        start_t = st.slider("Reconstruction start timestep", min_value=1, max_value=scheduler.T - 1, value=150)

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    model = load_model(checkpoint_path, str(device))
    st.success(f"Loaded checkpoint on {device.type.upper()}.")

    tab_generate, tab_reconstruct, tab_forward = st.tabs(["Generate", "Reconstruct", "Forward Diffusion"])

    with tab_generate:
        st.subheader("Generate from pure noise")
        if st.button("Generate images", type="primary"):
            with st.spinner("Sampling from noise..."):
                generated, intermediates = sample_from_noise(model, scheduler, device, generated_images, preview_steps, seed)

            st.pyplot(render_image_row(generated, "Generated Images"), clear_figure=True)
            st.pyplot(render_step_grid(intermediates, "Reverse Diffusion Steps"), clear_figure=True)

    with tab_reconstruct:
        st.subheader("Reconstruction demo")
        uploaded = st.file_uploader("Upload a target image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded target", use_container_width=True)

            if st.button("Run reconstruction"):
                with st.spinner("Noising and denoising the uploaded image..."):
                    reference, noisy, reconstructed, intermediates = reconstruct_from_image(
                        model, scheduler, device, image, start_t, preview_steps, seed
                    )

                col1, col2, col3 = st.columns(3)
                col1.image(to_image(reference), caption="Target", use_container_width=True)
                col2.image(to_image(noisy), caption=f"Noisy start (t={start_t})", use_container_width=True)
                col3.image(to_image(reconstructed), caption="Reconstruction", use_container_width=True)

                psnr, ssim = compute_metrics(reference, reconstructed)
                metric1, metric2 = st.columns(2)
                metric1.metric("PSNR", f"{psnr:.2f} dB")
                metric2.metric("SSIM", f"{ssim:.4f}" if ssim is not None else "Install scikit-image")
                if not METRICS_AVAILABLE:
                    st.info("scikit-image is not installed in this environment, so SSIM is not available yet. Install the requirements file to enable it.")
                st.pyplot(render_step_grid(intermediates, "Reverse Diffusion During Reconstruction"), clear_figure=True)

    with tab_forward:
        st.subheader("Forward diffusion visualization")
        uploaded_forward = st.file_uploader("Upload an image for noising preview", type=["png", "jpg", "jpeg"], key="forward_uploader")
        if uploaded_forward is not None:
            image = Image.open(uploaded_forward).convert("RGB")
            st.image(image, caption="Original image", use_container_width=True)

            if st.button("Show forward steps"):
                x0 = to_tensor(image, NATIVE_IMG_SIZE).to(device)
                steps = make_step_schedule(scheduler.T, preview_steps)
                step_map = {0: x0.detach().cpu()}
                for step in steps:
                    noisy, _ = scheduler.add_noise(x0, torch.tensor([step], device=device, dtype=torch.long))
                    step_map[step] = noisy.detach().cpu()

                st.pyplot(render_step_grid(step_map, "Forward Diffusion Steps"), clear_figure=True)


if __name__ == "__main__":
    main()

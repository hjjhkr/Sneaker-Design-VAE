import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class BetaVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_log_var = nn.Linear(4096, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 4096)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        flat = torch.flatten(encoded, start_dim=1)
        mu = self.fc_mu(flat)
        log_var = self.fc_log_var(flat)
        z = self.reparameterize(mu, log_var)
        decoded = self.fc_decode(z).view(-1, 256, 4, 4)
        reconstruction = self.decoder(decoded)
        return reconstruction, mu, log_var


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pick_checkpoint(project_root: Path) -> Path:
    ckpt_dir = project_root / "artifacts" / "checkpoints"
    improved = ckpt_dir / "beta_vae_improved.pt"
    baseline = ckpt_dir / "beta_vae_baseline.pt"

    if improved.exists():
        return improved
    if baseline.exists():
        return baseline
    raise FileNotFoundError(
        "No checkpoint found. Expected artifacts/checkpoints/beta_vae_improved.pt "
        "or artifacts/checkpoints/beta_vae_baseline.pt."
    )


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    latent_dim = int(checkpoint.get("latent_dim", 16))

    model = BetaVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, latent_dim


def decode_png_bytes(model: BetaVAE, device: torch.device, latent_dim: int, dims_values):
    z = torch.zeros(1, latent_dim, device=device)

    for idx, value in enumerate(dims_values[:latent_dim]):
        z[0, idx] = float(value)

    with torch.no_grad():
        decoded = model.fc_decode(z).view(-1, 256, 4, 4)
        generated = model.decoder(decoded)

    image = generated[0].detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def build_handler(html_path: Path, model: BetaVAE, device: torch.device, latent_dim: int, checkpoint_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, path: Path):
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send_html(html_path)
                return
            if parsed.path == "/api/meta":
                self._send_json(
                    200,
                    {
                        "checkpoint": checkpoint_name,
                        "latent_dim": latent_dim,
                    },
                )
                return

            self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/generate":
                self._send_json(404, {"error": "Not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(data.decode("utf-8"))
                dims = payload.get("dims", [0.0] * latent_dim)

                if not isinstance(dims, list):
                    self._send_json(400, {"error": "dims must be a list of numeric values"})
                    return

                if len(dims) == 0:
                    dims = [0.0] * latent_dim

                if len(dims) > latent_dim:
                    self._send_json(
                        400,
                        {"error": f"dims length ({len(dims)}) exceeds latent_dim ({latent_dim})"},
                    )
                    return

                png_bytes = decode_png_bytes(model, device, latent_dim, dims)
                png_base64 = base64.b64encode(png_bytes).decode("utf-8")

                self._send_json(
                    200,
                    {
                        "image_base64": png_base64,
                        "checkpoint": checkpoint_name,
                        "latent_dim": latent_dim,
                        "dims_used": len(dims),
                    },
                )
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, _fmt, *_args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve interactive sneaker design HTML with model inference API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    project_root = resolve_project_root()
    html_path = project_root / "web" / "custom_design_interactive.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Missing HTML file: {html_path}")

    checkpoint_path = pick_checkpoint(project_root)
    device = pick_device()
    model, latent_dim = load_model(checkpoint_path, device)
    handler = build_handler(
        html_path=html_path,
        model=model,
        device=device,
        latent_dim=latent_dim,
        checkpoint_name=checkpoint_path.name,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

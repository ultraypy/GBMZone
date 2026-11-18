"""GO/KEGG + H&E + RNA GraphSAGE; compatible with all four saved bundles."""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

METHODS = ('pathway_he_tran', 'he_tran', 'pathway_tran', 'tran')
ALIASES = dict(full_half='pathway_he_tran', he_rna_quarter='he_tran',
               path_rna_third='pathway_tran', rna_zero='tran')
FILENAMES = dict(pathway_he_tran='pathway+he+tran.pt', he_tran='he+tran.pt',
                 pathway_tran='pathway+tran.pt', tran='tran.pt')


def canonical_method(name):
    name = ALIASES.get(name, name)
    if name not in METHODS:
        raise ValueError(f'Unknown architecture: {name}')
    return name


def normgraph(a):
    """Row-normalized existing within-tissue graph; isolated nodes retain self."""
    a = a.astype(np.float32).tocsr()
    if a.shape[0] != a.shape[1] or not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError('Adjacency must be square, finite and nonnegative')
    d = np.asarray(a.sum(1)).ravel()
    a = a + sp.diags((d == 0).astype(np.float32))
    return (sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a).astype(np.float32)


def tgraph(a, device):
    a = a.tocoo()
    return torch.sparse_coo_tensor(
        torch.as_tensor(np.array([a.row, a.col]), device=device),
        torch.as_tensor(a.data, device=device), a.shape).coalesce()


class FourModel(nn.Module):
    def __init__(self, method, mask, path_sd, groups, hidden=24):
        super().__init__()
        self.method = canonical_method(method)
        self.use_path = self.method in ('pathway_he_tran', 'pathway_tran')
        self.use_image = self.method in ('pathway_he_tran', 'he_tran')
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or len(groups) != mask.shape[1]:
            raise ValueError('Gene/pathway membership dimensions disagree')
        if self.use_path and not (mask.sum(0) > 0).all():
            raise ValueError('Every pathway must have measured members')
        self.register_buffer('mask', torch.as_tensor(mask))
        self.register_buffer('path_sd', torch.as_tensor(path_sd, dtype=torch.float32))
        # Historical buffer required for strict loading of existing weights.
        self.register_buffer('met', torch.as_tensor(np.array(groups) == 'metabolic'))
        self.s1 = nn.Linear(mask.shape[0], hidden)
        self.n1 = nn.Linear(mask.shape[0], hidden, bias=False)
        self.s2 = nn.Linear(hidden, hidden)
        self.n2 = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, 3)
        self.drop = nn.Dropout(.1)
        if self.use_path:
            self.path_logits = nn.Parameter(torch.zeros(mask.shape))
            self.register_buffer('is_go', torch.as_tensor(np.array(groups) == 'GO_BP'))
            ng = int(self.is_go.sum()); nk = int((~self.is_go).sum())
            self.go_self = nn.Linear(ng, hidden, bias=False)
            self.go_neigh = nn.Linear(ng, hidden, bias=False)
            self.kegg_self = nn.Linear(nk, hidden, bias=False)
            self.kegg_neigh = nn.Linear(nk, hidden, bias=False)
            self.path_gates = nn.Parameter(torch.tensor([-1., -1.]))
        if self.use_image:
            self.image_self = nn.Linear(512, hidden, bias=False)
            self.image_neigh = nn.Linear(512, hidden, bias=False)
            self.image_gate = nn.Parameter(torch.tensor(-1.))

    def forward(self, x, a, image=None, nx=None, ni=None):
        h = self.s1(x) + self.n1(torch.sparse.mm(a, x) if nx is None else nx)
        if self.use_path:
            weights = torch.softmax(self.path_logits.masked_fill(~self.mask, -1e9), dim=0)
            score = (x @ weights) / self.path_sd
            neighbor = torch.sparse.mm(a, score)
            g = torch.sigmoid(self.path_gates)
            h = h + g[0] * (self.go_self(score[:, self.is_go]) + self.go_neigh(neighbor[:, self.is_go]))
            h = h + g[1] * (self.kegg_self(score[:, ~self.is_go]) + self.kegg_neigh(neighbor[:, ~self.is_go]))
        if self.use_image:
            if image is None:
                raise ValueError('This architecture requires H&E features')
            h = h + torch.sigmoid(self.image_gate) * (
                self.image_self(image) + self.image_neigh(torch.sparse.mm(a, image) if ni is None else ni))
        h = self.drop(torch.nn.functional.silu(h))
        h = self.drop(torch.nn.functional.silu(self.s2(h) + self.n2(torch.sparse.mm(a, h))))
        return self.out(h)


class Block(nn.Module):
    def __init__(self, inc, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out); self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = nn.Sequential(nn.Conv2d(inc, out, 1, stride, bias=False), nn.BatchNorm2d(out)) if stride != 1 or inc != out else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + residual)


class ResNet18(nn.Module):
    """Exact state-key-compatible frozen ImageNet encoder; no torchvision needed."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64); self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(Block(64, 64), Block(64, 64))
        self.layer2 = nn.Sequential(Block(64, 128, 2), Block(128, 128))
        self.layer3 = nn.Sequential(Block(128, 256, 2), Block(256, 256))
        self.layer4 = nn.Sequential(Block(256, 512, 2), Block(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def load_encoder(bundle, device='cpu'):
    net = ResNet18()
    net.load_state_dict(bundle['image_encoder_state'], strict=True)
    net.fc = nn.Identity()
    return net.eval().requires_grad_(False).to(device)


def extract_he_features(rgb, xy, pixels_per_um, encoder, device='cpu', batch_size=128):
    """220um square -> 224x224, white outside image, frozen ResNet18 -> 512D.

    xy must already be in this raw RGB image's coordinate system. No registration
    is estimated here. RGB must be an HxWx3 uint8 array, never a label overlay.
    """
    device = torch.device(device)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or pixels_per_um <= 0:
        raise ValueError('Expected RGB uint8 and positive spatial scale')
    height, width = rgb.shape[:2]
    if min(height, width) < 2 or len(xy) == 0:
        raise ValueError('Image and coordinates must be nonempty')
    if not np.isfinite(xy).all():
        raise ValueError('Nonfinite image coordinates')
    image = torch.tensor(rgb.copy().transpose(2, 0, 1), dtype=torch.float32, device=device)[None] / 255
    axis = torch.linspace(-.5, .5, 224, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    base = torch.stack([xx, yy], dim=-1)[None]
    scale = torch.tensor([2 / (width - 1), 2 / (height - 1)], device=device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    sd = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    centers = torch.as_tensor(xy, dtype=torch.float32, device=device)
    encoder.eval(); output = []
    with torch.inference_mode():
        for first in range(0, len(xy), batch_size):
            c = centers[first:first + batch_size]
            grid = (c[:, None, None, :] + base * (float(pixels_per_um) * 220)) * scale - 1
            patch = torch.nn.functional.grid_sample((image - 1).expand(len(c), -1, -1, -1), grid, mode='bilinear', padding_mode='zeros', align_corners=True) + 1
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                output.append(encoder((patch - mean) / sd).float().cpu().numpy())
    return np.concatenate(output)


def load_fold(path, fold=0, device='cpu'):
    """Load only trusted, user-owned PyTorch bundles."""
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    ck = bundle['folds'][fold]
    model = FourModel(ck['method'], ck['mask'], ck['path_sd'], ck['groups'], ck['hidden'])
    model.load_state_dict(ck['state'], strict=True)
    return model.to(device).eval(), ck['preprocessing'], bundle

import torch
import torch.nn.functional as F
import math


def rgb2lum(image):
    image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
    return image[:, :, :, None]


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)


def lerp(a, b, l):
    return (1 - l) * a + l * b


def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):

        def activation(x):
            if initial is not None:
                bias = torch.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return torch.tanh(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)


def DarkChannel(im):
    """
    适用于PyTorch张量的暗通道计算
    输入参数：
        im - 形状为 (B, C, H, W) 的浮点型张量，取值范围[0,1]
    返回值：
        dc - 形状为 (B, 1, H, W) 的暗通道图
    """
    # 沿通道维度取最小值（假设输入为RGB顺序）
    dc, _ = torch.min(im, dim=1, keepdim=True)  # dim=1对应通道维度
    return dc

def AtmLight(im):
    """
    PyTorch版本大气光估计
    参数：
        im - 输入图像张量 (B, C, H, W)
        dark - 暗通道张量 (B, 1, H, W)
    返回：
        A - 大气光估计 (B, C)
    """
    B, C, H, W = im.shape
    dark = DarkChannel(im)
    dark_flat = dark.view(B, 1, -1)
    im_flat = im.contiguous().view(B, C, -1)
    _, indices = torch.topk(dark_flat, k=max(math.floor(H * W / 1000), 1), dim=2)
    expanded_indices = indices.expand(-1, C, -1)
    # 收集对应像素值
    selected_pixels = torch.gather(im_flat, dim=2, index=expanded_indices)
    return selected_pixels.mean(dim=2).unsqueeze(-1).unsqueeze(-1)

def DarkIcA(im, Atm):
    """
    PyTorch版本暗通道归一化计算
    参数：
        im - 输入图像张量，形状为 (B, C, H, W)
        A - 大气光张量，形状为 (B, C)
    返回：
        dark_ica - 归一化后的暗通道，形状为 (B, 1, H, W)
    """
    # 添加极小值防止除零错误
    eps = 1e-8
    # 归一化各通道 (自动广播)
    im_normalized = im / (Atm + eps)  # 广播到 (B, C, H, W)
    # 计算归一化后的暗通道
    dark_ica = DarkChannel(im_normalized)

    return dark_ica


class Filter:

    def __init__(self, cfg):
        self.cfg = cfg
        # self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))

        # Specified in child classes
        self.begin_filter_parameter = None
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def extract_parameters(self, features):
        return features[:,
               self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

    # Should be implemented in child classes
    def filter_param_regressor(self, features):
        assert False

    def process(self, img, param):
        assert False

    # Apply the whole filter with masking
    def apply(self, img, img_features=None, specified_parameter=None):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            filter_parameters = specified_parameter

        low_res_output = self.process(img, filter_parameters)

        return low_res_output, filter_parameters

class UsmFilter(Filter):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1
        self.sigma = 4  # 可调整的模糊强度参数

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def make_gaussian_1d_kernel(self, sigma, device):
        """生成一维高斯核"""
        radius = 3 * sigma  # 覆盖99.7%能量
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()  # 归一化
        return kernel.view(1, 1, 1, -1)  # [1,1,1,kernel_size]

    def process(self, img, param):
        """
        Args:
            img: 输入图像 tensor，形状 [B,C,H,W]
            param: 锐化强度参数 tensor，形状 [B,1]
        Returns:
            img_out: 锐化后的图像 [B,C,H,W]
        """
        # 生成一维高斯核
        kernel_1d = self.make_gaussian_1d_kernel(self.sigma, img.device)  # [1,1,1,k]
        kernel_1d = kernel_1d.repeat(3, 1, 1, 1)

        # 水平方向卷积
        pad_h = (kernel_1d.shape[-1] - 1) // 2
        blurred_h = F.conv2d(
            img,
            kernel_1d,
            stride=1,
            padding=(0, pad_h),  # 仅水平方向填充
            groups=img.shape[1]  # 并行处理所有通道
        )

        # 垂直方向卷积
        blurred = F.conv2d(
            blurred_h,
            kernel_1d.transpose(-1, -2),  # 转置为垂直核 [1,1,k,1]
            stride=1,
            padding=(pad_h, 0),  # 仅垂直方向填充
            groups=img.shape[1]
        )

        # 锐化处理
        param = param.view(-1, 1, 1, 1)  # 扩展维度 [B,1,1,1]
        img_out = img + (img - blurred) * param

        return img_out


class DefogFilter(Filter): #Defog_param is in [Defog_range]

    def __init__(self,  cfg):
        Filter.__init__(self,  cfg)
        self.short_name = 'DF'
        self.begin_filter_parameter = cfg.defog_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.defog_range)(features)

    def process(self, img, param):
        # 输入维度验证
        assert img.dim() == 4, "Input image must be 4D tensor (B, C, H, W)"
        assert param.dim() == 2 and param.shape[1] == 1, "Param must be (B, 1)"

        # 批处理计算各分量
        B, C, H, W = img.shape
        defog_A = AtmLight(img)
        IcA = DarkIcA(img, defog_A)

        param_4d = param.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        tx = 1 - param_4d * IcA  # 广播后 [B, 1, H, W]
        tx_3ch = tx.expand(-1, 3, -1, -1)  # 扩展至 [B, C, H, W]

        # tx_clamped = torch.clamp(tx_3ch, min=0.01)
        # numerator = img - defog_A  # 广播减法 [B, C, H, W]
        # result = numerator / tx_clamped + defog_A

        tx_clamped = torch.clamp(tx_3ch, min=0.1, max=0.99)
        numerator = img - defog_A * (1 - tx_clamped)  # 改进公式
        result = numerator / tx_clamped

        return result

class GammaFilter(Filter):  # gamma_param is in [-gamma_range, gamma_range]

    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        gamma_range = torch.tensor(self.cfg.gamma_range, dtype=torch.float32, device=features.device)
        log_gamma_range = torch.log(gamma_range)
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param):
        mi = torch.tensor(0.0001, dtype=torch.float32, device=img.device)
        return torch.pow(torch.maximum(img, mi), param.view(-1, 1, 1, 1))


class ImprovedWhiteBalanceFilter(Filter):

    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = 3

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32, device=features.device)

        masked_features = features * mask  # shape remains (B, 3, H, W)

        scaled_features = torch.tanh(masked_features) * log_wb_range
        color_scaling = torch.exp(scaled_features)

        weights = torch.tensor([0.27, 0.67, 0.06], device=features.device)

        luminance = torch.sum(color_scaling * weights, dim=1, keepdim=True)
        color_scaling = color_scaling / (luminance + 1e-5)

        return color_scaling

    def process(self, img, param):
        param = param.unsqueeze(-1).unsqueeze(-1)
        return img * param


class ToneFilter(Filter):

    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.curve_steps = 8
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param

        self.num_filter_parameters = 8

    def filter_param_regressor(self, features):
        tone_curve = features.view(-1, 1, self.cfg.curve_steps)[:, None, None, :]
        return tanh_range(*self.cfg.tone_curve_range)(tone_curve)

    def process(self, img, param):
        tone_curve_sum = param.sum(dim=4) + 1e-30
        # 创建索引张量 [curve_steps]
        i_vals = torch.arange(self.cfg.curve_steps, device=img.device, dtype=img.dtype) / self.cfg.curve_steps

        # 扩展维度用于广播 [B,C,H,W,1] - [curve_steps] => [B,C,H,W,curve_steps]
        img_expanded = img.unsqueeze(-1)

        # 向量化计算
        clamped = torch.clamp(img_expanded - i_vals, 0, 1.0 / self.cfg.curve_steps)
        total_image = (clamped * param).sum(dim=4)

        total_image = total_image * (self.cfg.curve_steps / tone_curve_sum)
        return total_image

class ContrastFilter(Filter):

    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param

        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        # return tf.sigmoid(features)
        return torch.tanh(features)

    def process(self, img, param):
        # 获取输入 img 的设备，确保常量与 img 在相同设备上
        device = img.device
        luminance = torch.minimum(torch.maximum(rgb2lum(img), torch.tensor(0.0, device=device)),
                                  torch.tensor(1.0, device=device))
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        # 计算对比度图像
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param.unsqueeze(-1).unsqueeze(-1).to(device))

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np

# 修改导入语句，处理VectorQuantizer2导入失败的情况
try:
    from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
except ImportError:
    print("警告：无法导入VectorQuantizer2，使用替代方案")
    # 方案1：尝试直接导入VectorQuantizer
    try:
        from taming.modules.vqvae.quantize import VectorQuantizer
        print("成功导入VectorQuantizer作为替代")
    except ImportError:
        # 方案2：如果仍然无法导入，我们可能需要定义一个简化版本的VectorQuantizer
        print("错误：无法导入VectorQuantizer，请确保正确安装了taming-transformers包")
        
        # 定义一个简化版VectorQuantizer用于应急
        class VectorQuantizer(torch.nn.Module):
            def __init__(self, n_e, e_dim, beta, remap=None, sane_index_shape=False):
                super().__init__()
                self.n_e = n_e
                self.e_dim = e_dim
                self.beta = beta
                self.embedding = torch.nn.Embedding(self.n_e, self.e_dim)
                self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
                
            def forward(self, z):
                # 简化版forward，仅用于使模型能够加载
                z_flattened = z.view(-1, self.e_dim)
                d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                    torch.matmul(z_flattened, self.embedding.weight.t())
                min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
                min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
                min_encodings.scatter_(1, min_encoding_indices, 1)
                z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
                loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
                z_q = z + (z_q - z).detach()
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
                return z_q, loss, (0, min_encodings, min_encoding_indices)
                
            def get_codebook_entry(self, indices, shape):
                min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
                min_encodings.scatter_(1, indices[:,None], 1)
                z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
                if shape is not None:
                    z_q = z_q.view(shape)
                    z_q = z_q.permute(0, 3, 1, 2).contiguous()
                return z_q

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.automatic_optimization = False  # 设置为手动优化模式
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        # 获取指定键的数据
        x = batch[k]
        
        # 处理不同的输入格式
        # 如果输入已经是[B, C, H, W]格式，直接返回
        if len(x.shape) == 4 and (x.shape[1] == 1 or x.shape[1] == 3):
            return x.float()
            
        # 如果是PET数据的[B, H, W]格式（单通道），添加通道维度
        if len(x.shape) == 3 and 'patient_name' in batch:
            x = x.unsqueeze(1)  # 变成[B, 1, H, W]
            return x.float()
            
        # 原始处理方式（[B, H, W, C] -> [B, C, H, W]）
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx):
        # 获取优化器
        optimizers = self.optimizers()
        # 检查optimizers是否是列表
        if isinstance(optimizers, list):
            opt_ae, opt_disc = optimizers
        else:
            # 如果只有一个优化器，那应该是自编码器的优化器
            opt_ae = optimizers
            opt_disc = None
        
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        # 训练自编码器
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train",
                                        predicted_indices=ind)
        self.manual_backward(aeloss)
        opt_ae.step()
        
        # 如果有判别器才训练判别器
        if opt_disc is not None:
            opt_disc.zero_grad()
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.manual_backward(discloss)
            opt_disc.step()
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return None

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        # Check if the loss has a discriminator
        if hasattr(self.loss, 'discriminator') and self.loss.discriminator is not None:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], []
        else:
            # 如果没有判别器，使用更明确的PyTorch Lightning兼容格式
            return ([opt_ae], [])  # 返回优化器列表和空的学习率调度器列表

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.automatic_optimization = False  # 设置为手动优化模式
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        # 获取指定键的数据
        x = batch[k]
        
        # 处理不同的输入格式
        # 如果输入已经是[B, C, H, W]格式，直接返回
        if len(x.shape) == 4 and (x.shape[1] == 1 or x.shape[1] == 3):
            return x.float()
            
        # 如果是PET数据的[B, H, W]格式（单通道），添加通道维度
        if len(x.shape) == 3 and 'patient_name' in batch:
            x = x.unsqueeze(1)  # 变成[B, 1, H, W]
            return x.float()
            
        # 原始处理方式（[B, H, W, C] -> [B, C, H, W]）
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        # 获取优化器
        optimizers = self.optimizers()
        # 检查optimizers是否是列表
        if isinstance(optimizers, list):
            opt_ae, opt_disc = optimizers
        else:
            # 如果只有一个优化器，那应该是自编码器的优化器
            opt_ae = optimizers
            opt_disc = None
        
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        # 训练编码器+解码器+logvar
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)
        opt_ae.step()
        
        # 如果有判别器才训练判别器
        if opt_disc is not None:
            opt_disc.zero_grad()
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.manual_backward(discloss)
            opt_disc.step()
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return None

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        # Check if the loss has a discriminator
        if hasattr(self.loss, 'discriminator') and self.loss.discriminator is not None:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], []
        else:
            # 如果没有判别器，使用更明确的PyTorch Lightning兼容格式
            return ([opt_ae], [])  # 返回优化器列表和空的学习率调度器列表

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


if __name__ == "__main__":
    import numpy as np
    from omegaconf import OmegaConf
    import sys
    import os
    
    # 添加项目根目录到PATH中以便导入dataset模块
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    print("测试AutoencoderKL模型...")
    
    def test_basic_functions():
        # 创建配置
        ddconfig = {
            "double_z": True,
            "z_channels": 32,
            "resolution": 256,
            "in_channels": 1,
            "out_ch": 1,
            "ch": 64,
            "ch_mult": [1, 2, 2],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        }
        
        lossconfig = {
            "target": "ldm.modules.losses.LPIPSWithDiscriminator",
            "params": {
                "disc_start": 50001,
                "kl_weight": 0.000001,
                "disc_weight": 0.5
            }
        }
        
        # 创建一个测试批次
        batch_size = 2
        img_size = 256
        test_img = torch.randn(batch_size, 1, img_size, img_size)
        
        try:
            # 初始化模型
            lossconfig_obj = OmegaConf.create(lossconfig)
            model = AutoencoderKL(ddconfig=ddconfig, 
                                  lossconfig=lossconfig_obj,
                                  embed_dim=3,
                                  image_key="image")
            
            # 将模型设置为评估模式
            model.eval()
            
            # 创建一个测试批次数据
            test_batch = {"image": test_img.permute(0, 2, 3, 1)}
            
            # 测试编码和解码
            with torch.no_grad():
                posterior = model.encode(test_img)
                z = posterior.sample()
                reconstructed = model.decode(z)
                
            # 打印结果
            print(f"原始图像形状: {test_img.shape}")
            print(f"编码结果形状: {z.shape}")
            print(f"重建图像形状: {reconstructed.shape}")
            
            # 测试log_images方法
            with torch.no_grad():
                log_dict = model.log_images(test_batch)
                
            print("log_images 输出字典包含以下键:")
            for key in log_dict.keys():
                print(f"  - {key}: {log_dict[key].shape}")
            
            print("基本功能测试成功!")
            return True
            
        except Exception as e:
            print(f"基本功能测试失败，错误: {e}")
            return False
    
    def test_with_pet_lowdose_data():
        print("\n正在测试与PET低剂量数据的兼容性...")
        try:
            # 尝试导入数据集
            from dataset import Datasetlow
            
            # 创建数据集实例
            dataroot = input("请输入数据根目录路径（包含pet_lowdose_test目录的路径）: ")
            if not dataroot or not os.path.exists(dataroot):
                print(f"路径不存在: {dataroot}")
                return False
                
            # 创建一个小型数据集
            dataset = Datasetlow(dataroot, "test", data_len=2)
            print(f"成功创建数据集，长度为: {len(dataset)}")
            
            # 获取一个样本
            if len(dataset) > 0:
                img_lr, img_hr, patient_name = dataset[0]
                print(f"获取样本 - 低分辨率图像形状: {img_lr.shape}")
                print(f"获取样本 - 高分辨率图像形状: {img_hr.shape}")
                print(f"获取样本 - 患者名称: {patient_name}")
                
                # 创建配置
                ddconfig = {
                    "double_z": True,
                    "z_channels": 32,
                    "resolution": 384,
                    "in_channels": 1,
                    "out_ch": 1,
                    "ch": 64,
                    "ch_mult": [1, 2, 2],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0
                }
                
                lossconfig = {
                    "target": "ldm.modules.losses.LPIPSWithDiscriminator",
                    "params": {
                        "disc_start": 50001,
                        "kl_weight": 0.000001,
                        "disc_weight": 0.5
                    }
                }
                
                # 初始化模型
                lossconfig_obj = OmegaConf.create(lossconfig)
                model = AutoencoderKL(ddconfig=ddconfig, 
                                      lossconfig=lossconfig_obj,
                                      embed_dim=3,
                                      image_key="hr")
                
                # 将模型设置为评估模式
                model.eval()
                
                # 扩展batch维度，创建一个包含单个样本的batch
                img_lr_batch = img_lr.unsqueeze(0)
                img_hr_batch = img_hr.unsqueeze(0)
                
                # 创建一个测试批次数据（适配AutoencoderKL.get_input方法的格式）
                test_batch = {
                    "lr": img_lr_batch,
                    "hr": img_hr_batch,
                    "patient_name": patient_name
                }
                
                # 添加一个用于PET数据的log_images方法
                original_log_images = model.log_images
                
                def patched_log_images(batch, **kwargs):
                    if isinstance(batch, dict) and "hr" in batch:
                        # 适配PET数据格式
                        # 创建新的batch格式
                        new_batch = {"image": batch["hr"].permute(0, 2, 3, 1)}
                        return original_log_images(new_batch, **kwargs)
                    else:
                        return original_log_images(batch, **kwargs)
                
                # 应用补丁
                model.log_images = patched_log_images
                
                # 测试log_images方法
                with torch.no_grad():
                    log_dict = model.log_images(test_batch)
                    
                print("使用PET数据的log_images输出字典包含以下键:")
                for key in log_dict.keys():
                    print(f"  - {key}: {log_dict[key].shape}")
                
                # 测试编码和解码
                with torch.no_grad():
                    posterior = model.encode(img_hr_batch)
                    z = posterior.sample()
                    reconstructed = model.decode(z)
                    
                # 打印结果
                print(f"原始高分辨率图像形状: {img_hr_batch.shape}")
                print(f"编码结果形状: {z.shape}")
                print(f"重建图像形状: {reconstructed.shape}")
                
                print("PET低剂量数据测试成功!")
                return True
                
            else:
                print("数据集为空")
                return False
                
        except Exception as e:
            print(f"PET低剂量数据测试失败，错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 运行测试
    test_basic_functions()
    
    # 询问用户是否要测试PET低剂量数据
    answer = input("\n是否要测试PET低剂量数据? (y/n): ")
    if answer.lower() == 'y':
        test_with_pet_lowdose_data()

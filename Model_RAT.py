import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MLP(nn.Module):

    def __init__(self, in_feat, h_feat=None, out_feat=None):
        super().__init__()
        
        self.fc1 = nn.Conv2d(in_channels=in_feat, out_channels=h_feat, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels=h_feat, out_channels=out_feat, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class RegionAttention(nn.Module):
    def __init__(self, n_feat, num_heads, bias=True):
        super(RegionAttention, self).__init__()
        self.num_heads = num_heads


        self.qkv = nn.Conv2d(n_feat, n_feat*3, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias) 
    
    def global_attention(self, q, k, v, mask):

        #mask shape: [b, l, l] 
        B, C, H, W = q.shape
        mask = repeat(mask, 'b l1 l2 -> b head l1 l2', head=self.num_heads)
        
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads) 
        attn = (q @ k.transpose(-2, -1).contiguous()) * (self.num_heads ** -0.5) #[b head l l]
        attn = attn + mask
        
        attn = attn.softmax(dim=-1) #[b head l l]

        out = (attn @ v) #[b head l c]
        
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', h=H, w=W) 
        
        return out

    def forward(self, x, mask):    

        b, c, h, w = x.shape 

        qkv = self.qkv(x) 
        q,k,v = qkv.chunk(3, dim=1)   
        out = self.global_attention(q, k, v, mask)
        
        out = self.project_out(out) 

        return out 

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=4):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class CAB(nn.Module):
    def __init__(self, num_features, reduction=4):
        super(CAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features//4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_features//4, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return self.module(x)



class WindowAttention(nn.Module):
    def __init__(self, n_feat, num_heads, window_size, bias=True):
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads 
        self.window_size = window_size
        # self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Linear(n_feat, n_feat * 3, bias=bias)
        self.proj_out = nn.Linear(n_feat, n_feat) 
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, C, H, W, C)
            window_size (int): window size
    
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, C, H, W = x.shape 
        x = rearrange(x, 'b c h w -> b h w c') 
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    
    def window_reverse(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image
    
        Returns:
            x: (B, C, H, W)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1) 
        x = rearrange(x, 'b h w c -> b c h w') 
        return x 
    
    def window_attention(self, x): 
        B_, N, C = x.shape 
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, heads, N, C//heads]
        attn = (q @ k.transpose(-2, -1).contiguous())* (self.num_heads ** -0.5)
        attn = self.softmax(attn) 
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C) 
        x = self.proj_out(x) 
        return x


    def forward(self, x):    
        
        B, C, H, W = x.shape 
        x_windows = self.window_partition(x, self.window_size) #[nW*B, window_size, window_size, C] 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # [nW*B, window_size*window_size, C] 
        
        attn_windows = self.window_attention(x_windows) 
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) 
        out = self.window_reverse(attn_windows, self.window_size, H, W) # [B, C, H, W]
        return out 




class BasicLayer(nn.Module):
    def __init__(self, n_feat, num_heads, window_size, mlp_ratio):
        super(BasicLayer, self).__init__() 
        
        self.norm_ra = LayerNorm2d(n_feat) 
        self.ra = RegionAttention(n_feat, num_heads) 
        self.lambda_ra = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

        self.norm_ra_mlp = LayerNorm2d(n_feat) 
        self.mlp_ra = MLP(n_feat, int(n_feat*mlp_ratio), n_feat) 
        
        self.norm_wa = LayerNorm2d(n_feat) 
        self.wa = WindowAttention(n_feat, num_heads, window_size) 
        self.lambda_wa = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)
        
        self.norm_wa_mlp = LayerNorm2d(n_feat) 
        self.mlp_wa = MLP(n_feat, int(n_feat*mlp_ratio), n_feat) 
        
        
        
        
    
    def forward(self, x, mask): 

        x_norm = self.norm_ra(x)
        x = x + self.lambda_ra*self.ra(x_norm, mask) 
        x = self.mlp_ra(self.norm_ra_mlp(x)) + x  
        
        x_norm = self.norm_wa(x)
        x = x + self.lambda_wa*self.wa(x_norm) 
        x = self.mlp_wa(self.norm_wa_mlp(x)) + x  

        return x


class Block(nn.Module):
    def __init__(self, n_layer, n_feat, num_heads, window_size, mlp_ratio):
        super(Block, self).__init__() 
        self.layers = nn.ModuleList([BasicLayer(n_feat, 
                                               num_heads,
                                               window_size,
                                               mlp_ratio)
                                     for i in range(n_layer)]) 
        
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, stride=1, bias=True)

    
    def forward(self, x, mask): 
        shotcut = x
        for layer in self.layers:
            x = layer(x, mask) 
        x = self.conv(x)
        return x + shotcut

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma




class RAT(nn.Module):

    def __init__(self, 
                 scale=1,
                 img_channel=1, 
                 width=64, 
                 middle_blk_num=12, 
                 enc_blk_nums=[2, 2], 
                 dec_blk_nums=[2, 2], 
                 loss_fun = None):
        
        super().__init__() 
        self.img_scale = scale
        self.mask_scale = 2**len(enc_blk_nums) 
        
        

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        
        n_layer = 3
        self.middle_blks = nn.ModuleList([Block(n_layer=n_layer, 
                                                   n_feat=chan, 
                                                   num_heads=8,
                                                   window_size=4,
                                                   mlp_ratio=2)
                                             for i in range(int(middle_blk_num//n_layer))]) 

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        # self.padder_size = 2 ** len(self.encoders) 
        self.loss_fun = loss_fun 
        
    @torch.no_grad()
    def correlation_mask_oneBYone(self, mask, value=-1000):
        b, c, h, w = mask.shape 
        device = mask.device 
        
        value = torch.Tensor([value]).type(torch.FloatTensor).to(device)
        out_mask = [] 

        for bi in range(b):
            mask_bi = mask[bi] #[C, H, W] 
            total_area=0 
            
            
            corr_bi = torch.zeros((1, h*w, h*w), device=device)
            for cls_i in range(int(mask_bi.max()+1)): 
                region = mask_bi==cls_i 
                area_i = torch.sum(region)

                if area_i>0:
                    region=region.type(torch.FloatTensor).to(device).unsqueeze(-1).view(c, -1, 1) #C, H*W, 1
                    
                    
                    corr_bi+= region @ region.transpose(-2, -1)
                    
                    total_area += area_i 
            if total_area!=(mask_bi.shape[-1]*mask_bi.shape[-2]):
                raise ValueError("Total Area Error!") 
            
            out_mask.append(corr_bi) 
        out_mask = torch.concat(out_mask, dim=0) # [B, H*W ,H*W] 
        out_mask = (1-out_mask)*value
        return out_mask 
    
    @torch.no_grad()
    def correlation_mask_batch(self, mask, value=-1000):
        b, c, h, w = mask.shape 
        device = mask.device 
        
        value = torch.Tensor([value]).type(torch.FloatTensor).to(device)
        out_mask = [] 

        for bi in range(b):
            mask_bi = mask[bi] #[1, H, W] 
            total_area=0 
            
            
            regions = []
            for cls_i in range(int(mask_bi.max()+1)): 
                region_i = mask_bi==cls_i 
                area_i = torch.sum(region_i)

                if area_i>0:
                    region_i=region_i.type(torch.FloatTensor).to(device).unsqueeze(-1).view(c, -1, 1) #1, H*W, 1
                    regions.append(region_i)
                    total_area += area_i 
            if total_area!=(mask_bi.shape[-1]*mask_bi.shape[-2]):
                raise ValueError("Total Area Error!") 
            
            
            regions = torch.concat(regions, dim=0) #[R, H*W, 1] 
            region_att = regions @ regions.transpose(-2, -1) #[R, H*W, H*W] 
            region_att = torch.sum(region_att, dim=0, keepdim=True)
            
            out_mask.append(region_att) 
        out_mask = torch.concat(out_mask, dim=0) # [B, R, H*W, 1] 
        
        out_mask = (1-out_mask)*value
        return out_mask 

    def forward(self, inp, mask_original, label=None): 
        if self.img_scale>1:
            inp = F.interpolate(inp, scale_factor=self.img_scale, mode='bilinear')

        mask = F.interpolate(mask_original.unsqueeze(1), scale_factor = 1/self.mask_scale, mode='nearest') 
        
        

        mask = self.correlation_mask_oneBYone(mask, -1000) 

        
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        shortcut = x 

        for blk in self.middle_blks:
            x = blk(x, mask) 
            
        x = x + shortcut

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        if label is None:
            return x 
        else:
            return self.loss_fun(x, label, mask_original) 
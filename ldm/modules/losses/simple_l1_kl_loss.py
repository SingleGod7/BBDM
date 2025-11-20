import torch
import torch.nn as nn

class L1KLoss(nn.Module):
    def __init__(self, kl_weight=1.0, logvar_init=0.0, l1_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.l1_weight = l1_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        
        # この損失関数は生成器の更新にのみ使用されるため、optimizer_idx == 0 の場合のみを処理します。
        # 実際には、呼び出し側が optimizer_idx を正しく処理することを期待します。
        # ここでは、LPIPSWithDiscriminator との互換性のために引数を維持します。
        if optimizer_idx != 0:
            # 判別器の更新の場合は何も返さないか、エラーを発生させることもできます。
            # 簡単のため、None を返します。
            return None, {}


        # L1 loss (reconstruction loss)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        # NLL loss (weighted L1 loss)
        # LPIPSWithDiscriminatorと同様に、logvarを使用してL1損失を重み付けします
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        else:
            weighted_nll_loss = nll_loss
            
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # For logging purposes, calculate unweighted nll_loss mean as well
        nll_loss_mean = torch.sum(nll_loss) / nll_loss.shape[0]
        # For logging purposes, calculate unweighted rec_loss mean as well
        rec_loss_mean = torch.sum(rec_loss) / rec_loss.shape[0]

        # KL divergence
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Total loss
        loss = self.l1_weight * weighted_nll_loss + self.kl_weight * kl_loss

        log = {
            f"{split}/total_loss": loss.clone().detach().mean(),
            f"{split}/logvar": self.logvar.detach(),
            f"{split}/kl_loss": kl_loss.detach().mean(),
            f"{split}/nll_loss": nll_loss_mean.detach().mean(), # log unweighted nll_loss
            f"{split}/rec_loss": rec_loss_mean.detach().mean(), # log unweighted rec_loss
        }
        
        return loss, log 
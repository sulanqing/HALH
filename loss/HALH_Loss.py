import torch.nn as nn

class HALH_Loss(nn.Module):
    """
    Loss function of ADSH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma,lambda0):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma
        self.lambda0 = lambda0
        # self.classify_loss=nn.CrossEntropyLoss()
        # self.classify_loss=nn.BCEWithLogitsLoss()

    def forward(self, F, B, S, omega):
        h=F @ B.t()
        g=B[omega, :]
        

        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        
        quantization_loss = ((F - B[omega, :]) ** 2).sum()
        balance_loss = F.mean(dim=1).pow(2).mean()
        
        loss1 = (hash_loss + self.gamma * quantization_loss) / (F.shape[0] * B.shape[0])
        loss = loss1 + self.lambda0*balance_loss

        return loss

    

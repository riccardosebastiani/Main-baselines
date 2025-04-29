from Data_Set import KirchhoffDataset
import torch
import torch.nn as nn
class KirchhoffLoss(torch.nn.Module):
    def __init__(self, plate: KirchhoffDataset):
        super(KirchhoffLoss, self).__init__()
        self.plate = plate

    def call(self, preds, xy):
        xy = xy['coords']
        x, y, t = xy[:, :, 0], xy[:, :, 1], xy[:, :, 2]
        preds = preds['model_out']
        L_f, L_b0, L_b2, L_err = self.plate.compute_loss(x, y, t, preds)
        return {'L_f': L_f, 'L_b0': L_b0, 'L_b2': L_b2, 'L_err': L_err}


class CustomVariable(nn.Module):
    def __init__(self, value, trainable=True):
        super(CustomVariable, self).__init__()
        self.value = nn.Parameter(torch.tensor(value), requires_grad=trainable)

    def forward(self):
        return self.value

    def assign(self, new_value):
        self.value.data = torch.tensor(new_value, dtype=self.data.dtype)

class KirchhoffLoss(nn.Module):
    def __init__(self, model, plate: KirchhoffDataset):
        super().__init__()
        self.plate = plate
        self.model = model
        self.w = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms - 1)]
        self.lambdas = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms)]

    def call(self, preds, xy):
        xy = xy['coords']
        x, y, t = xy[:, :, 0], xy[:, :, 1], xy[:, :, 2]
        preds = preds['model_out']
        eps = 0.01

        losses = [torch.mean(loss) for loss in self.plate.compute_loss(x, y, t, preds)]
        log_sigma_value = self.model.log_sigma
        w = [torch.exp(log_sigma_value[i]) for i in range(len(log_sigma_value))]
        l = [1 / (2 * (eps ** 2 + wi ** 2)) * loss + torch.log(eps ** 2 + wi ** 2) for loss, wi in zip(losses[:-1], w)]
        l.append(losses[-1])
        a = [1 / (2 * (eps ** 2 + wi ** 2)) for wi in w]
        self.lambdas = [ai.detach().requires_grad_(False) for ai in a]

        return {'L_f': losses[0], 'L_b0': losses[1], 'L_b2': losses[2], 'L_err': losses[3]}

class KirchhoffMetric(nn.Module):
    def __init__(self, plate):
        super(KirchhoffMetric, self).__init__()
        self.plate = plate
        self.L_f_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_err_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b0_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b2_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

    def update_state(self, xy, y_pred):
        xy = xy['coords']
        y_pred = y_pred['model_out']
        x, y, t = xy[:, :, 0], xy[:, :, 1], xy[:, :, 2]

        L_f, L_b0, L_b2, L_err = self.plate.compute_loss(x, y, t, y_pred)
        self.L_f_mean.data = torch.mean(L_f)
        self.L_err_mean.data = torch.mean(L_err)
        self.L_b0_mean.data = torch.mean(L_b0)
        self.L_b2_mean.data = torch.mean(L_b2)

    def reset_state(self):
        self.L_f_mean.data = torch.zeros(1)
        self.L_err_mean.data = torch.zeros(1)
        self.L_b0_mean.data = torch.zeros(1)
        self.L_b2_mean.data = torch.zeros(1)

    def result(self):
        return {'L_f': self.L_f_mean.item(),
                'L_err': self.L_err_mean.item(),
                'L_b0': self.L_b0_mean.item(),
                'L_b2': self.L_b2_mean.item()}


class LambdaMetric(nn.Module):
    def __init__(self, loss):
        super(LambdaMetric, self).__init__()
        self.loss = loss
        self.L_f_lambda_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_err_lambda_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b0_lambda_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b2_lambda_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

    def update_state(self, xy, y_pred, sample_weight=None):
        L_f_lambda, L_b0_lambda, L_b2_lambda, L_err_lambda = self.loss.lambdas

        self.L_f_lambda_mean.data = torch.tensor(L_f_lambda.item())
        self.L_err_lambda_mean.data = torch.tensor(L_err_lambda.item())
        self.L_b0_lambda_mean.data = torch.tensor(L_b0_lambda.item())
        self.L_b2_lambda_mean.data = torch.tensor(L_b2_lambda.item())

    def reset_state(self):
        self.L_f_lambda_mean.data = torch.zeros(1)
        self.L_err_lambda_mean.data = torch.zeros(1)
        self.L_b0_lambda_mean.data = torch.zeros(1)
        self.L_b2_lambda_mean.data = torch.zeros(1)

    def result(self):
        return {'L_f': self.L_f_lambda_mean.item(),
                'L_err': self.L_err_lambda_mean.item(),
                'L_b0': self.L_b0_lambda_mean.item(),
                'L_b2': self.L_b2_lambda_mean.item()}
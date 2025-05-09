import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
from Data_Set import compute_derivatives


class Sine(nn.Module):
    def __init__(self, w0=5.0):
        super(Sine, self).__init__()

        self.w0 = nn.Parameter(torch.tensor(w0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

''' 
class Sine(nn.Module):
    def __init(self):
        super().__init__()
#prova 0.06 poi
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(5 * input)
#9 + 256 neuroni
'''
#now
#6,7,8,9,5, 6.5

class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'silu': (nn.SiLU(), init_weights_xavier, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features, bias=True), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features, bias=True), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features, bias=True)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations

def min_max_normalization(data, min_val, max_val):
    return 2 * (data - min_val) / (max_val - min_val) - 1
def time_norm(time):
    t = min_max_normalization(time, 0, torch.max(time))
    return t*100
class PINNet(nn.Module):
    '''Architecture used by Raissi et al. 2019.'''

    def __init__(self, out_features=1, type='tanh', in_features=3, mode='mlp'):
        super().__init__()
        self.mode = mode
        self.n_terms = 5
        self.log_sigma = nn.Parameter(torch.zeros(self.n_terms - 1), requires_grad=True)

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=4,
                           hidden_features=400, outermost_linear=True, nonlinearity=type,
                           weight_init=None)
        print(self)

    def forward(self, model_input):

        coords = model_input['coords']
        x, y, t = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
        x = x[..., None].requires_grad_(True)
        y = y[..., None].requires_grad_(True)
        t = t[..., None].requires_grad_(True)

        o = self.net(torch.cat((x, y, t), dim=-1))
        dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt = compute_derivatives(x, y, t, o)
        output = torch.cat((o, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt), dim=-1)
        return {'model_in': coords, 'model_out': output}


class MultiScalePINNet(nn.Module):
    def __init__(self, scales=[1,2,4,8], out_features=1, type='sine', in_features=3):
        super(MultiScalePINNet, self).__init__()
        self.scales = scales
        self.n_terms = 5
        self.log_sigma = nn.Parameter(torch.zeros(self.n_terms - 1), requires_grad=True)
        self.models = nn.ModuleList([
            FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=5,
                    hidden_features=64, outermost_linear=True, nonlinearity=type,
                    weight_init=None)
            for _ in scales
        ])
        self.combined_layer = nn.Linear(len(scales) * out_features, out_features)

    def forward(self, model_input):

        coords = model_input['coords']
        x, y, t = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
        x = x[..., None]
        y = y[..., None]
        t = t[..., None]
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        inputs = [torch.cat((x, y, t), dim=-1) for _ in self.scales]
        outputs = [model(input) for model, input in zip(self.models, inputs)]
        combined_output = torch.cat(outputs, dim=-1)
        final_output = self.combined_layer(combined_output)

        dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt = compute_derivatives(x, y, t, final_output)
        output = torch.cat((final_output, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt), dim=-1)
        return {'model_in': coords, 'model_out': output}

########################
# Initialization methods

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 25, np.sqrt(6 / num_input) / 25)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

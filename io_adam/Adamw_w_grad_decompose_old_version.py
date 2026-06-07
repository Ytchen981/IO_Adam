import torch
from torch import nn
from torch.optim import Optimizer, AdamW
from typing import Any, Iterable, Callable, Iterable, Tuple
from functools import wraps
import math
import warnings
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from transformers.utils.versions import require_version

def non_zero_min(x: Tensor):
    return torch.where(x == 0, torch.tensor(float('inf'), device=x.device), x).min().item()

class GradBuffer:
    def __init__(self, in_dim: int, out_dim:int, beta1: float = None, beta2: float = None, p: float = 2, max_queue_length: int = 16, w_momentum: bool = True, clear_final: bool = False, merge_final: bool = False, tight_bound: bool = False) -> None:
        self.full = False
        self.w_momentum = w_momentum
        if self.w_momentum:
            self.exp_avg = torch.zeros((out_dim, in_dim))
            self.beta1 = beta1
        else:
            self.exp_avg = None
        self.beta2 = beta2
        self.p = p
        self.q = 1 / (1 - 1/ p)
        self.v_in_queue = torch.zeros(max_queue_length, in_dim)
        self.v_out_queue = torch.zeros(max_queue_length, out_dim)
        self.max_queue_length = max_queue_length
        self.pointer = 0
        self.clear_final = clear_final
        self.merge_final = merge_final
        self.clear_flag = False
        self.tight_bound = tight_bound
        if tight_bound:
            self.input_min = None
            self.input_max = None
            self.grad_output_min = None
            self.grad_output_max = None
        #self.exp_avg_sq = torch.zeros((out_dim, in_dim))
        #self.count = 0


    @torch.no_grad
    def update(self, input, grad_output):
        if self.w_momentum:
            if input.device != self.exp_avg.device:
                self.exp_avg = self.exp_avg.to(input.device)
            #self.exp_avg_sq = self.exp_avg_sq.to(self.exp_avg.device)
            self.exp_avg.add_(grad_output.t().mm(input), alpha=1-self.beta1)
        else:
            self.exp_avg = grad_output.t().mm(input)
        if input.device != self.v_out_queue.device:
            self.v_in_queue = self.v_in_queue.to(input.device)
            self.v_out_queue = self.v_out_queue.to(input.device)

        if self.clear_flag and self.full:
            if self.merge_final:
                next_pointer = (self.pointer + 1) % self.max_queue_length
                self.v_out_queue[next_pointer].add_(self.v_out_queue[self.pointer])
                self.v_in_queue[next_pointer].add_(self.v_in_queue[self.pointer])
                #self.v_out_queue[next_pointer].add_(self.v_out_queue[self.pointer])
                #self.v_in_queue[next_pointer].add_(self.v_in_queue[self.pointer])
            self.v_in_queue[self.pointer].mul_(0.)
            self.v_out_queue[self.pointer].mul_(0.)
            self.clear_flag = False

        self.v_out_queue[self.pointer].add_(grad_output.pow(2).sum(dim=0), alpha=1-self.beta2)
        self.v_in_queue[self.pointer].add_(input.pow(2).sum(dim=0), alpha=1-self.beta2)

        if self.tight_bound:
            if self.input_min is None:
                self.input_min = non_zero_min(input.abs())
                self.input_max = input.abs().max().item()
            else:
                self.input_min = min(self.input_min, non_zero_min(input.abs()))
                self.input_max = max(self.input_max, input.abs().max().item())
            if self.grad_output_min is None:
                self.grad_output_min = non_zero_min(grad_output.abs())
                self.grad_output_max = grad_output.abs().max().item()
            else:
                self.grad_output_min = min(self.grad_output_min, non_zero_min(grad_output.abs()))
                self.grad_output_max = max(self.grad_output_max, grad_output.abs().max().item())

        #self.count += input.size(0)
        #self.exp_avg_sq.add_(grad_output.t().mm(input).pow(2), alpha=1-self.beta2)

    @torch.no_grad
    def avg_decay(self):
        if self.w_momentum:
            self.exp_avg.mul_(self.beta1)
        else:
            self.exp_avg = None
        self.v_out_queue.mul_(self.beta2)
        self.v_in_queue.mul_(self.beta2)

    def to(self, *args, **kwargs):
        self.exp_avg = self.exp_avg.to(*args, **kwargs)
        #self.exp_avg_sq = self.exp_avg_sq.to(*args, **kwargs)
        self.v_in_queue = self.v_in_queue.to(*args, **kwargs)
        self.v_out_queue = self.v_out_queue.to(*args, **kwargs)

    @property
    def data(self):
        if self.pointer == self.max_queue_length - 1:
            self.full = True
        if self.tight_bound:
            holder_constant = ((self.input_max * self.grad_output_max)**2 - (self.input_min * self.grad_output_min)**2) / (2 * (self.input_max * self.grad_output_max - self.input_min * self.grad_output_min) * math.sqrt((self.grad_output_max * self.grad_output_min)*(self.input_max * self.input_min)))
            self.v_in_queue[self.pointer].mul_(math.sqrt((1 + 1/holder_constant) / 2))
            self.v_out_queue[self.pointer].mul_(math.sqrt((1 + 1/holder_constant) / 2))
        self.pointer = (self.pointer + 1) % self.max_queue_length
        if self.clear_final:
            self.clear_flag = True

        #return self.exp_avg, self.exp_avg_sq
        #return self.exp_avg, self.grad_output_sq, self.input_sq
        '''if self.w_momentum:
            return self.exp_avg, self.v_out_queue.t().mm(self.v_in_queue).sqrt()
        else:
            return self.grad_output.t().mm(self.input), self.v_out_queue.t().mm(self.v_in_queue).sqrt()'''
        return self.exp_avg, self.v_out_queue.t().mm(self.v_in_queue).sqrt()

class DecomposedLinear(nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.decompose_linear_function = None

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def prepare_decompose_linear(self, Buffer: GradBuffer):
        class Decomposed_LinearFunction(torch.autograd.Function):
            # Note that both forward and backward are @staticmethods
            @staticmethod
            # bias is an optional argument
            def forward(ctx, input, weight, bias=None):
                #print(input.shape)
                ctx.save_for_backward(input, weight, bias)
                output = torch.matmul(input, weight.t())
                if bias is not None:
                    output += bias.unsqueeze(0).expand_as(output)
                return output

            # This function has only a single output, so it gets only one gradient
            @staticmethod
            def backward(ctx, grad_output):
                # This is a pattern that is very convenient - at the top of backward
                # unpack saved_tensors and initialize all gradients w.r.t. inputs to
                # None. Thanks to the fact that additional trailing Nones are
                # ignored, the return statement is simple even when the function has
                # optional inputs.
                input, weight, bias = ctx.saved_tensors
                grad_input = grad_bias = None

                # These needs_input_grad checks are optional and there only to
                # improve efficiency. If you want to make your code simpler, you can
                # skip them. Returning gradients for inputs that don't require it is
                # not an error.
                #if ctx.needs_input_grad[0]:
                    #grad_input = grad_output.mm(weight)
                #if ctx.needs_input_grad[1]:
                    #grad_weight = grad_output.t().mm(input)
                #if bias is not None and ctx.needs_input_grad[2]:
                    #grad_bias = grad_output.sum(0)
                if ctx.needs_input_grad[0]:
                    grad_input = torch.matmul(grad_output, weight)
                if bias is not None:
                    grad_bias = grad_output.sum(0)

                Buffer.update(input.reshape(-1, input.size(-1)), grad_output.reshape(-1, grad_output.size(-1)))
                return grad_input, None, grad_bias
        self.decompose_linear_function = Decomposed_LinearFunction

    def disable_decompose(self):
        self.decompose_linear_function = None

    def forward(self, input: Tensor) -> Tensor:
        if self.decompose_linear_function is None:
            return F.linear(input, self.weight, self.bias)
        else:
            return self.decompose_linear_function.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class AdamW_Decomposed(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            p:float = 1.1,
            max_queue_length:int = 16,
            w_momentum:bool = True,
            clear_final:bool = False,
            merge_final:bool = False,
            tight_bound: bool = False
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        self.p = p
        self.max_queue_length = max_queue_length
        self.w_momentum = w_momentum
        self.clear_final = clear_final
        self.merge_final = merge_final
        self.tight_bound = tight_bound
        super().__init__(params, defaults)

    def __param_to_module__(self, model):
        # get the mapping from parameters to all Linear Module
        param_to_module = {}
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters(recurse=False):  # 只看当前 module 的参数
                    full_name = f"{module_name}.{name}" if module_name else name
                    if "weight" in full_name:
                        param_to_module[param] = (module, module_name)
                    else:
                        param_to_module[param] = (None, None)
        return param_to_module

    def replace_module(self, model):
        param_to_module = self.__param_to_module__(model)
        for group in self.param_groups:
            new_params = []
            replaced = set()
            beta1, beta2 = group["betas"]
            for p in group['params']:
                if p in param_to_module.keys():
                    module, module_name = param_to_module[p]
                    if module is not None and id(module) not in replaced:
                        if p in self.state.keys():
                            state = self.state.pop(p)
                        else:
                            state = dict()
                        new_module = DecomposedLinear(module.in_features, module.out_features, bias=(module.bias is not None), dtype=model.dtype).to(model.device)
                        new_module.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_module.bias.data.copy_(module.bias.data)
                        grad_buffer = GradBuffer(module.in_features, module.out_features, beta1, beta2, self.p, self.max_queue_length, self.w_momentum, self.clear_final, self.merge_final, self.tight_bound)
                        state["grad_buffer"] = grad_buffer
                        new_module.prepare_decompose_linear(grad_buffer)

                        parent = model
                        if '.' in module_name:
                            path = module_name.split('.')
                            for p in path[:-1]:
                                parent = getattr(parent, p)
                            setattr(parent, path[-1], new_module)
                        else:
                            setattr(model, module_name, new_module)
                        new_params += [p for p in new_module.parameters() if p.requires_grad]
                        self.state[new_module.weight] = state
                        replaced.add(id(module))
                    else:
                        pass
                else:
                    new_params.append(p)
            group["params"] = new_params


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                if p.grad is None:
                    state = self.state[p]
                    if "grad_buffer" in state.keys():
                        if "step" not in state:
                            state["step"] = 0
                        state["step"] += 1
                        exp_avg, exp_avg_sq = state["grad_buffer"].data
                        beta1, beta2 = group["betas"]
                        #exp_avg.div_(grad_output_sq.t().mm(input_sq).sqrt().add_(group["eps"]))
                        denom = exp_avg_sq.add_(group["eps"])
                        step_size = group["lr"]
                        if group["correct_bias"]:  # No bias correction for Bert
                            if self.w_momentum:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                            else:
                                bias_correction1 = 1.0
                            #step = min(self.max_queue_length, state["step"])
                            bias_correction2 = 1.0 - beta2 ** state["step"]
                            step_size = step_size * bias_correction2 / bias_correction1
                        p.addcdiv_(exp_avg, denom, value=-step_size)
                        #p.add_(exp_avg, alpha=-step_size)
                        state["grad_buffer"].avg_decay()
                    else:
                        continue
                else:
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    if "step" not in state:
                        state["step"] = 0

                    # State initialization
                    if "exp_avg" not in state:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    # compute norm gradient
                    norm_grad = exp_avg / denom

                    p.add_(norm_grad, alpha=-step_size)



        return loss

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    model = AutoModelForCausalLM.from_pretrained("/home_new/chenyiting/RoPE_angle/model/Llama-2-7b-hf",torch_dtype=torch.float16, trust_remote_code=True)
    optimizer = AdamW_Decomposed(model.parameters(), lr=1e-3)
    import time
    tick = time.time()
    optimizer.replace_module(model)
    tock = time.time()
    print(tock - tick)
    optimizer.test_queue()
    '''for name, module in model.named_modules():
        if isinstance(module, DecomposedLinear):
            print(name)'''


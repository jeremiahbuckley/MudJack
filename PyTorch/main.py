from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
import time
import traceback

app = FastAPI()

def test_functions():
    results = {}
    functions_to_test = {
        'torch.argsort': test_argsort,
        'torch.optim.Adamax': test_optim_adamax,
        'torch.fliplr': test_fliplr,
        'torch.broadcast_tensors': test_broadcast_tensors,
        'torch.nn.AdaptiveAvgPool3d': test_adaptive_avg_pool3d,
        'torch.addr': test_addr,
        'torch.cat': test_cat,
        'torch.optim.LBFGS': test_optim_lbfgs,
        'torch.triangular_solve': test_triangular_solve,
        'torch.nn.Module.state_dict': test_module_state_dict,
        'torch.nn.Module.zero_grad': test_module_zero_grad,
        'torch.sum': test_sum,
        'torch.Tensor.is_same_size': test_is_same_size,
        'torch.nn.KLDivLoss': test_kl_div_loss,
        'torch.nn.LSTMCell': test_lstm_cell,
        'torch.moveaxis': test_moveaxis,
        'torch.nn.functional.dropout': test_functional_dropout,
        'torch.lt': test_lt,
        'torch.autograd.Variable': test_autograd_variable,
        'torch.utils.data.Subset': test_data_subset,
        'torch.nn.Sequential': test_sequential,
        'torch.multinomial': test_multinomial,
        'torch.nn.Linear': test_linear,
        'torch.nn.BCEWithLogitsLoss': test_bce_with_logits_loss,
        'torch.diag': test_diag,
        'torch.distributions.von_mises.VonMises': test_von_mises,
        'torch.zeros': test_zeros,
        'torch.round': test_round,
        'torch.arange': test_range,  # torch.range is deprecated
        'torch.autograd.functional.jvp': test_jvp,
        'torch.linalg.matrix_rank': test_matrix_rank,
        'torch.nn.GELU': test_gelu,
        'torch.Tensor.to': test_tensor_to,
        'torch.nn.Transformer': test_transformer,
        'torch.bitwise_not': test_bitwise_not,
        'torch.nn.Conv3d': test_conv3d,
        'torch.slogdet': test_slogdet,
        'torch.optim.lr_scheduler.ExponentialLR': test_exponential_lr,
        'torch.utils.data.Dataset': test_dataset,
        'torch.utils.data.ConcatDataset': test_concat_dataset,
        'torch.nn.Module.register_parameter': test_register_parameter,
        'torch.cuda': test_cuda,
        'torch.nn.Conv2d': test_conv2d,
    }

    for func_name, func in functions_to_test.items():
        func_result = {'error': None, 'average_time': None}
        try:
            # Warm-up run
            func()

            # Timing over 10 runs
            total_time = 0.0
            for _ in range(10):
                start_time = time.time()
                func()
                end_time = time.time()
                total_time += (end_time - start_time)
            func_result['average_time'] = total_time / 10
        except Exception as e:
            func_result['error'] = traceback.format_exc()
        results[func_name] = func_result

    return results

# Test functions
def test_argsort():
    x = torch.randn(1000)
    torch.argsort(x)

def test_optim_adamax():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adamax(model.parameters())
    data = torch.randn(10)
    target = torch.randn(1)
    output = model(data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

def test_fliplr():
    x = torch.randn(3, 4)
    torch.fliplr(x)

def test_broadcast_tensors():
    a = torch.randn(3, 1)
    b = torch.randn(1, 4)
    torch.broadcast_tensors(a, b)

def test_adaptive_avg_pool3d():
    input = torch.randn(1, 64, 8, 9, 10)
    m = torch.nn.AdaptiveAvgPool3d((5, 7, 9))
    m(input)

def test_addr():
    x = torch.randn(3, 3)
    vec1 = torch.randn(3)
    vec2 = torch.randn(3)
    torch.addr(x, vec1, vec2)

def test_cat():
    tensors = [torch.randn(2, 3), torch.randn(2, 3)]
    torch.cat(tensors, dim=0)

def test_optim_lbfgs():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.LBFGS(model.parameters())
    data = torch.randn(10)
    target = torch.randn(1)

    def closure():
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        return loss

    optimizer.step(closure)

def test_triangular_solve():
    b = torch.randn(3, 3)
    A = torch.randn(3, 3)
    torch.triangular_solve(b, A)

def test_module_state_dict():
    model = torch.nn.Linear(10, 1)
    state_dict = model.state_dict()

def test_module_zero_grad():
    model = torch.nn.Linear(10, 1)
    model.zero_grad()

def test_sum():
    x = torch.randn(1000)
    torch.sum(x)

def test_is_same_size():
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    result = x.size() == y.size()

def test_kl_div_loss():
    loss = torch.nn.KLDivLoss()
    input = torch.randn(3, requires_grad=True)
    target = torch.randn(3)
    output = loss(input.log_softmax(dim=0), target.softmax(dim=0))
    output.backward()

def test_lstm_cell():
    rnn = torch.nn.LSTMCell(10, 20)
    input = torch.randn(3, 10)
    hx = torch.randn(3, 20)
    cx = torch.randn(3, 20)
    hx, cx = rnn(input, (hx, cx))

def test_moveaxis():
    x = torch.randn(2, 3, 4)
    torch.moveaxis(x, 0, -1)

def test_functional_dropout():
    x = torch.randn(10)
    torch.nn.functional.dropout(x, p=0.5, training=True)

def test_lt():
    x = torch.randn(10)
    y = torch.randn(10)
    torch.lt(x, y)

def test_autograd_variable():
    x = torch.autograd.Variable(torch.randn(10), requires_grad=True)
    y = x * 2
    y.backward(torch.ones_like(x))

def test_data_subset():
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))
    subset = torch.utils.data.Subset(dataset, indices=range(50))

def test_sequential():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )
    input = torch.randn(5, 10)
    output = model(input)

def test_multinomial():
    weights = torch.randn(10).abs()
    torch.multinomial(weights, num_samples=5, replacement=True)

def test_linear():
    m = torch.nn.Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)

def test_bce_with_logits_loss():
    loss = torch.nn.BCEWithLogitsLoss()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(input, target)
    output.backward()

def test_diag():
    x = torch.randn(3, 3)
    torch.diag(x)

def test_von_mises():
    concentration = torch.tensor([1.0])
    loc = torch.tensor([0.0])
    von_mises = torch.distributions.VonMises(loc, concentration)
    sample = von_mises.sample()

def test_zeros():
    torch.zeros(3, 3)

def test_round():
    x = torch.randn(10)
    torch.round(x)

def test_range():
    torch.arange(0, 10)

def test_jvp():
    def func(x):
        return x ** 2

    x = torch.tensor(1.0, requires_grad=True)
    v = torch.tensor(1.0)
    torch.autograd.functional.jvp(func, x, v)

def test_matrix_rank():
    x = torch.randn(10, 10)
    torch.linalg.matrix_rank(x)

def test_gelu():
    m = torch.nn.GELU()
    input = torch.randn(10)
    output = m(input)

def test_tensor_to():
    x = torch.randn(10)
    x.to(torch.float64)

def test_transformer():
    model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    out = model(src, tgt)

def test_bitwise_not():
    x = torch.tensor([0, 1, 2], dtype=torch.int32)
    torch.bitwise_not(x)

def test_conv3d():
    m = torch.nn.Conv3d(16, 33, 3, stride=2)
    input = torch.randn(20, 16, 10, 50, 100)
    output = m(input)

def test_slogdet():
    a = torch.randn(3, 3)
    sign, logdet = torch.slogdet(a)

def test_exponential_lr():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    optimizer.step()
    scheduler.step()

def test_dataset():
    class CustomDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.randn(10), torch.tensor(0)

    dataset = CustomDataset()

def test_concat_dataset():
    dataset1 = torch.utils.data.TensorDataset(torch.randn(100, 10))
    dataset2 = torch.utils.data.TensorDataset(torch.randn(50, 10))
    concat_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

def test_register_parameter():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.register_parameter('param', torch.nn.Parameter(torch.randn(10)))

    model = MyModule()

def test_cuda():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x = torch.randn(10, device=device)
    else:
        x = torch.randn(10)

def test_conv2d():
    m = torch.nn.Conv2d(16, 33, 3, stride=2)
    input = torch.randn(20, 16, 50, 100)
    output = m(input)

@app.get("/test")
async def run_tests():
    results = test_functions()
    return JSONResponse(content=results)


#!/usr/bin/env python3
"""Synthetic FP32 SnakeBeta parity, state preservation and autograd fallback."""
from _common import Report, graph_ms, import_error, load_gpu, metrics, parser


def main():
    cli = parser(__doc__, seed=2026)
    args = cli.parse_args()
    torch = load_gpu(args, cli, bf16=False)
    try:
        from yue2.modeling_vae import SnakeBeta
        from yue2.vae_kernels import fused_snake_beta
    except ImportError as exc:
        import_error(cli, exc)

    with Report(args, torch, "vae_snake_beta", shape=[2, 16, 1001],
                dtype="float32", atol=1e-6, rtol=1e-6) as report:
        for varied_parameters in (False, True):
            act = SnakeBeta(16).cuda().eval()
            if varied_parameters:
                with torch.no_grad():
                    act.alpha.uniform_(-0.5, 0.5)
                    act.beta.uniform_(-0.5, 0.5)
            state = {key: value.detach().clone() for key, value in act.state_dict().items()}
            for strided in (False, True):
                storage = torch.randn(2, 16, 1001 * (2 if strided else 1), device="cuda") * 3
                x = storage[..., ::2] if strided else storage
                with torch.inference_mode():
                    act._fused_snake_impl = None
                    reference = act(x)
                    stock_ms = graph_ms(torch, lambda: act(x), args)
                    act._fused_snake_impl = fused_snake_beta
                    actual = act(x)
                    numerical = metrics(torch, actual, reference)
                    report.add(varied_parameters=varied_parameters, strided_input=strided,
                               numerical=numerical, passed=False)
                    saved = report.data["rows"][-1]
                    assert numerical["finite"], numerical
                    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
                    fused_ms = graph_ms(torch, lambda: act(x), args)
                assert set(state) == set(act.state_dict())
                for key, value in act.state_dict().items():
                    assert torch.equal(value, state[key]), key

                def forbidden_fused(*unused):
                    raise AssertionError("inference-only kernel entered an autograd/training/CPU path")

                # A throwing sentinel proves fallback, rather than just relying
                # on a finite gradient that could have come from another path.
                act._fused_snake_impl = forbidden_fused
                differentiable = x.detach().clone().requires_grad_(True)
                act(differentiable).sum().backward()
                assert differentiable.grad is not None
                assert bool(differentiable.grad.isfinite().all())
                act.train()
                with torch.inference_mode():
                    torch.testing.assert_close(act(x), reference, atol=0, rtol=0)
                act.eval()
                saved.update(passed=True, state_unchanged=True,
                             autograd_and_training_fallback=True,
                             stock_ms=stock_ms, fused_ms=fused_ms)
                report.save()
            cpu_act = act.cpu()
            cpu_x = torch.randn(2, 16, 1001)
            with torch.inference_mode():
                fallback = cpu_act(cpu_x)
                cpu_act._fused_snake_impl = None
                assert torch.equal(fallback, cpu_act(cpu_x))
        assert len(report.data["rows"]) == 4
        report.data["cpu_fallback_equal_values"] = True


if __name__ == "__main__":
    main()

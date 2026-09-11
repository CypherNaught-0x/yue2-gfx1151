#!/usr/bin/env python3
"""Synthetic AR GEMV/RMSNorm numerical diagnostics and graph microbenchmarks."""
from _common import Report, graph_ms, import_error, load_gpu, metrics, parser


def linear_numerics(torch, actual, vendor, fp32):
    rounded = fp32.bfloat16()
    upper = torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).float()
    lower = torch.nextafter(rounded, torch.full_like(rounded, float("-inf"))).float()
    ulp = torch.maximum((upper - rounded.float()).abs(), (lower - rounded.float()).abs())
    error = (actual.float() - fp32).abs()
    vendor_error = (vendor.float() - fp32).abs()
    bound = 0.501 * ulp + 1e-4
    return {
        "max_abs_fp32": error.max().item(),
        "mean_abs_fp32": error.mean().item(),
        "vendor_max_abs_fp32": vendor_error.max().item(),
        "vendor_mean_abs_fp32": vendor_error.mean().item(),
        "max_excess_over_half_ulp": (error - 0.5 * ulp).max().item(),
        "round_mismatch": (actual != rounded).float().mean().item(),
        "vendor_round_mismatch": (vendor != rounded).float().mean().item(),
        "finite": bool(actual.isfinite().all() and vendor.isfinite().all() and fp32.isfinite().all()),
        "rounding_envelope_passed": bool((error <= bound).all()),
    }


def main():
    cli = parser(__doc__, seed=2026)
    cli.add_argument("--include-vocab", action="store_true", help="also allocate the original [184704,2048] vocabulary matrix and FP32 reference conversion (multi-GiB peak)")
    cli.add_argument("--quick", action="store_true", help="batch 1, first linear shape, default launch only; retain both RMS sizes")
    cli.add_argument("--numerics-only", action="store_true", help="omit graph timing, retain all numerical gates")
    args = cli.parse_args()
    torch = load_gpu(args, cli)
    try:
        import torch.nn.functional as F
        from yue2.ar_ops import linear, rms_norm
    except ImportError as exc:
        import_error(cli, exc)

    batches = (1,) if args.quick else (1, 4)
    shapes = [(4096, 2048)] if args.quick else [(4096, 2048), (12288, 2048), (2048, 6144)]
    if args.include_vocab:
        shapes.append((184704, 2048))
    launches = ((4, 4),) if args.quick else ((1, 4), (4, 4), (4, 8), (8, 4), (8, 8), (16, 8))
    with Report(args, torch, "ar_ops", batches=batches, shapes=shapes, launches=launches,
                timing=not args.numerics_only, linear_atol=0.003, linear_rtol=0.008,
                rounding_envelope="0.501 * local BF16 ULP + 1e-4",
                rms_atol=0, rms_rtol=0) as report, torch.inference_mode():
        for batch in batches:
            for n, k in shapes:
                x = torch.randn(batch, 1, k, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                vendor = F.linear(x, weight)
                # This is a vendor FP32 reduction oracle, not an exact-real dot.
                fp32 = F.linear(x.float(), weight.float())
                vendor_ms = None if args.numerics_only else graph_ms(torch, lambda: F.linear(x, weight), args)
                for block_m, num_warps in launches:
                    def kernel():
                        return linear(x, weight, block_m=block_m, num_warps=num_warps)

                    actual = kernel()
                    numerical = linear_numerics(torch, actual, vendor, fp32)
                    # Save diagnostics even when a numerical assertion fails.
                    row = dict(op="linear", batch=batch, n=n, k=k, block_m=block_m,
                               num_warps=num_warps, numerical=numerical, passed=False)
                    report.add(**row)
                    saved = report.data["rows"][-1]
                    assert numerical["finite"], numerical
                    assert numerical["rounding_envelope_passed"], numerical
                    torch.testing.assert_close(actual.float(), fp32, atol=0.003, rtol=0.008)
                    saved.update(passed=True, vendor_ms=vendor_ms,
                                 triton_ms=None if args.numerics_only else graph_ms(torch, kernel, args))
                    report.save()
                del weight, x, vendor, fp32, actual
            for n, heads in ((128, 16), (2048, 1)):
                for strided in (False, True):
                    # Strided rows exercise fused-QKV gaps without changing N.
                    storage = torch.randn(batch, 1, heads, n * (2 if strided else 1), device="cuda", dtype=torch.bfloat16)
                    x = storage[..., :n]
                    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)

                    def reference_norm():
                        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * weight

                    reference = reference_norm()
                    actual = rms_norm(x, weight, 1e-6)
                    numerical = metrics(torch, actual, reference)
                    report.add(op="rms_norm", batch=batch, n=n, heads=heads,
                               strided_rows=strided, numerical=numerical, passed=False)
                    saved = report.data["rows"][-1]
                    assert numerical["finite"], numerical
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                    # Equal tensor values, not a statement about signed-zero bits.
                    saved.update(passed=True, equal_values=True,
                                 vendor_ms=None if args.numerics_only else graph_ms(torch, reference_norm, args),
                                 triton_ms=None if args.numerics_only else graph_ms(torch, lambda: rms_norm(x, weight, 1e-6), args))
                    report.save()
        expected_rows = len(batches) * (len(shapes) * len(launches) + 4)
        assert len(report.data["rows"]) == expected_rows
        report.data["expected_rows"] = expected_rows


if __name__ == "__main__":
    main()

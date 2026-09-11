#!/usr/bin/env python3
"""Synthetic YuE2 BF16 single-token GQA: correctness before graph timing."""
from _common import Report, capture, graph_ms, import_error, load_gpu, metrics, parser


def main():
    cli = parser(__doc__, seed=123, repeats=40)
    cli.add_argument("--quick", action="store_true", help="batch 1; lengths 1/257; block 256 (smoke subset, not the original sweep)")
    cli.add_argument("--paired", action="store_true", help="also test the optional two-query-head shared-KV kernel")
    args = cli.parse_args()
    torch = load_gpu(args, cli)
    try:
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from yue2.ar_triton import decode_attention
    except ImportError as exc:
        import_error(cli, exc)

    batches = (1,) if args.quick else (1, 2, 4)
    lengths = (1, 257) if args.quick else (1, 127, 256, 257, 2048, 8192)
    blocks = (256,) if args.quick else (128, 256, 512)
    paired_modes = (False, True) if args.paired else (False,)
    capacity = 8192
    with Report(args, torch, "ar_attention", capacity=capacity, batches=batches,
                lengths=lengths, blocks=blocks, paired_modes=paired_modes,
                atol=0.002, rtol=0.02, baseline="explicit public SDPA MATH") as report, torch.inference_mode():
        for batch in batches:
            q = torch.randn(batch, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(batch, capacity, 8, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            for length in lengths:
                host_lengths = [max(1, length - row * 37) for row in range(batch)]
                used = torch.tensor(host_lengths, device="cuda", dtype=torch.int32)
                mask = (torch.arange(capacity, device="cuda")[None, :] < used[:, None])[:, None, None, :]

                def baseline():
                    # A fixed-capacity bool-masked SDPA baseline, NOT FlashAttention.
                    with sdpa_kernel(SDPBackend.MATH):
                        return F.scaled_dot_product_attention(
                            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                            attn_mask=mask, dropout_p=0.0, is_causal=False,
                            enable_gqa=True).transpose(1, 2)

                reference = baseline()
                assert bool(reference.isfinite().all()), "nonfinite SDPA reference"
                sdpa_ms = graph_ms(torch, baseline, args)
                for paired in paired_modes:
                    for block in blocks:
                        def kernel():
                            return decode_attention(q, k, v, used, block=block, paired=paired)

                        actual = kernel()
                        torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.02)
                        numerical = metrics(torch, actual, reference)
                        # Capture-time host constants must not determine visibility.
                        graph, output = capture(torch, kernel, args.warmup)
                        used.fill_(1)
                        graph.replay()
                        torch.testing.assert_close(output[:, 0], v[:, 0].repeat_interleave(2, dim=1), atol=0.002, rtol=0.02)
                        used.copy_(torch.tensor(host_lengths, device="cuda", dtype=torch.int32))
                        graph.replay()
                        torch.testing.assert_close(output, reference, atol=0.002, rtol=0.02)
                        del graph, output

                        # Poison all unused slots; don't ask masked SDPA to be a
                        # NaN oracle (0 * NaN is NaN in its fallback matmul).
                        poisoned_k, poisoned_v = k.clone(), v.clone()
                        for row, count in enumerate(host_lengths):
                            poisoned_k[row, count:] = float("nan")
                            poisoned_v[row, count:] = float("nan")

                        def poisoned_kernel():
                            return decode_attention(q, poisoned_k, poisoned_v, used, block=block, paired=paired)

                        poisoned = poisoned_kernel()
                        torch.testing.assert_close(poisoned, reference, atol=0.002, rtol=0.02)
                        poison_graph, poison_output = capture(torch, poisoned_kernel, args.warmup)
                        poison_graph.replay()
                        torch.testing.assert_close(poison_output, reference, atol=0.002, rtol=0.02)
                        used.fill_(1)
                        poison_graph.replay()
                        torch.testing.assert_close(poison_output[:, 0], v[:, 0].repeat_interleave(2, dim=1), atol=0.002, rtol=0.02)
                        used.copy_(torch.tensor(host_lengths, device="cuda", dtype=torch.int32))
                        del poison_graph, poison_output, poisoned, poisoned_k, poisoned_v
                        report.add(batch=batch, lengths=host_lengths, block=block, paired=paired,
                                   numerical=numerical, updated_length_replay=True,
                                   future_nan_eager_and_graph=True,
                                   triton_ms=graph_ms(torch, kernel, args), sdpa_math_ms=sdpa_ms,
                                   passed=True)
        expected_rows = len(batches) * len(lengths) * len(blocks) * len(paired_modes)
        assert len(report.data["rows"]) == expected_rows
        report.data["expected_rows"] = expected_rows


if __name__ == "__main__":
    main()

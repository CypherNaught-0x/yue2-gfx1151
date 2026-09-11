#!/usr/bin/env python3
"""Tiny random-model AR graph integration, row isolation and CFG contracts."""
from _common import Report, import_error, load_gpu, parser


def main():
    cli = parser(__doc__, seed=146)
    cli.add_argument("--linear", choices=("torch", "triton"), default="torch")
    cli.add_argument("--norm", choices=("torch", "triton"), default="torch")
    cli.add_argument("--fused", action="store_true", help="concatenate QKV and gate/up projections")
    args = cli.parse_args()
    torch = load_gpu(args, cli)
    try:
        from yue2.cuda_graph import GraphAR
        from yue2.modeling_yue2 import StaticKVCache, YuE2Config, YuE2ForCausalLM
    except ImportError as exc:
        import_error(cli, exc)
    options = dict(attention_backend="triton", linear_backend=args.linear,
                   norm_backend=args.norm, fuse_projections=args.fused)
    with Report(args, torch, "ar_graph_correctness", options=options,
                decode_atol=0.008, decode_rtol=0.04, timing=False) as report, torch.inference_mode():
        model = YuE2ForCausalLM(YuE2Config(
            hidden_size=256, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            vocab_size=256, max_position_embeddings=1024,
            max_latent_frames=1024)).eval().to(device="cuda", dtype=torch.bfloat16)
        for batch in (1, 2, 4):
            prefixes = [list(range(1, 129 - row * 23)) for row in range(batch)]
            caches = [StaticKVCache(2, 1, 2, len(prefix) + 8, 64, torch.bfloat16, "cuda") for prefix in prefixes]
            expected = torch.cat([
                model(torch.tensor([prefix], device="cuda"), past_key_values=cache,
                      use_cache=True, logits_to_keep=1).logits[:, -1]
                for prefix, cache in zip(prefixes, caches)])
            graph = GraphAR(model, prefixes, 8, independent_batch=True, **options)
            try:
                torch.testing.assert_close(graph.prefill(), expected, atol=0, rtol=0)
                for keys, values in zip(graph.keys, graph.values):
                    for row, prefix in enumerate(prefixes):
                        # Leave the imminent current-token slot alone; every
                        # remaining future slot must stay invisible until written.
                        keys[row, len(prefix) + 1:] = float("nan")
                        values[row, len(prefix) + 1:] = float("nan")
                maximum = 0.0
                for step in range(7):
                    tokens = torch.arange(batch, device="cuda") + 140 + step
                    expected = torch.cat([
                        model(token.reshape(1, 1), past_key_values=cache,
                              use_cache=True, logits_to_keep=1).logits[:, -1]
                        for token, cache in zip(tokens, caches)])
                    actual = graph.step(tokens)
                    assert bool(actual.isfinite().all())
                    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.04)
                    maximum = max(maximum, (actual.float() - expected.float()).abs().max().item())
                    assert graph.positions.tolist() == [len(prefix) + step + 1 for prefix in prefixes]
                    assert graph.tokens[:, 0].tolist() == tokens.tolist()
                report.add(batch=batch, decode_steps=7, max_abs=maximum,
                           prefill_equal_values=True, independent_positions=True,
                           future_nan_excluded=True, passed=True)
            finally:
                graph.close()
        graph = GraphAR(model, [[2, 3], [4]], 3, **options)
        try:
            graph.prefill()
            try:
                graph.step(torch.tensor([2, 3], device="cuda"))
            except ValueError:
                pass
            else:
                raise AssertionError("CFG accepted per-row tokens without independent_batch opt-in")
            graph.step(7)
            assert graph.tokens.tolist() == [[7], [7]]
            report.add(contract="legacy scalar-token CFG", passed=True)
        finally:
            graph.close()
        assert len(report.data["rows"]) == 4


if __name__ == "__main__":
    main()

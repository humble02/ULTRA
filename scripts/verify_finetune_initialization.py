#!/usr/bin/env python3
"""Check distilled-to-RL policy migration and prior-only gradients on CPU."""
import contextlib
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import torch
import yaml
from learning import ultra_models, ultra_network_builder_obj_v2, ultra_network_builder_obj_v3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "output/student_prior_fixed/g1_student_vae/nn/g1_student_vae_00010000.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/reproduction/finetune_from_10000")
    parser.add_argument("--expected-epoch", type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["epoch"] == args.expected_epoch

    def build(kind, builder_class):
        params = yaml.safe_load((ROOT / f"ultra/data/cfg/train/rlg/g1_student_{kind}.yaml").read_text())["params"]
        builder = builder_class()
        builder.load(params["network"])
        return ultra_models.ModelUltraContinuous(builder).build({
            "actions_num": 29, "input_shape": (1496,), "num_seqs": 1, "value_size": 1,
        })

    with (out / "initialization_build.log").open("w") as log, contextlib.redirect_stdout(log):
        distilled = build("vae", ultra_network_builder_obj_v2.UltraBuilder)
        rl = build("finetune", ultra_network_builder_obj_v3.UltraBuilder)
    distilled.load_state_dict(saved["model"], strict=True)
    current = rl.state_dict()
    weights = {key: value for key, value in saved["model"].items()
               if key in current and current[key].shape == value.shape
               and not (".critic" in key or ".value" in key)}
    result = rl.load_state_dict(weights, strict=False)
    assert all(any(s in key for s in (".critic", ".value", ".rl_finetune_film"))
               for key in result.missing_keys), result.missing_keys
    dropped = sorted(set(saved["model"]) - set(weights))
    assert all(".critic" in key or ".value" in key for key in dropped), dropped
    assert not result.unexpected_keys
    distilled.eval()
    rl.eval()
    obs = torch.randn(5, 1496) * .1
    obs[:, -242:] = 1
    teacher = torch.randn(5, 4052) * .1
    noise = torch.randn(5, 64)
    base = dict(is_train=False, obs=obs, prev_actions=None, rnn_states=None,
                teacher_obs=teacher, vae_noise=noise, skip_critic=True,
                with_encoder=False)
    with torch.no_grad():
        before = distilled({**base, "with_vae": False})["mus"]
        after = rl({**base, "with_vae": True})["mus"]
    error = float((before - after).abs().max())
    assert error < 1e-6, error
    calls = [0]
    def reject(module, inputs):
        calls[0] += 1
        raise AssertionError("RL prior invoked privileged encoder")
    hook = rl.a2c_network.encoder_net.register_forward_pre_hook(reject)
    with torch.no_grad():
        changed = rl({**base, "with_vae": True, "teacher_obs": teacher + 100})["mus"]
    assert torch.equal(after, changed), "Prior action depends on teacher observation"
    rl.zero_grad(set_to_none=True)
    prediction = rl({**base, "with_vae": True})["mus"]
    prediction.square().mean().backward()
    hook.remove()
    nonzero = [name for name, p in rl.named_parameters()
               if p.grad is not None and bool(p.grad.abs().sum() > 0)]
    assert nonzero and any("prior_" in name for name in nonzero), nonzero
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in rl.parameters())
    assert all(p.grad is None for name, p in rl.named_parameters() if not p.requires_grad)
    frozen = [name for name, p in rl.named_parameters() if not p.requires_grad]
    assert any("encoder_net" in name for name in frozen)
    assert any("actor_mlp" in name for name in frozen)
    rl.zero_grad(set_to_none=True)
    result = rl({**base, "with_vae": True, "skip_critic": False})
    assert torch.isfinite(result["values"]).all()
    result["values"].square().mean().backward()
    critic_grad = [name for name, p in rl.named_parameters()
                   if p.grad is not None and bool(p.grad.abs().sum() > 0)]
    assert critic_grad and all("critic" in name for name in critic_grad), critic_grad
    report = {"passed": True, "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_epoch": saved["epoch"], "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "initial_deploy_action_max_abs_error": error,
              "all_actor_weights_preserved": True, "loaded_state_entries": len(weights),
              "new_or_reinitialized_state_entries": list(rl.load_state_dict(weights, strict=False).missing_keys),
              "removed_state_entries": dropped, "prior_privileged_encoder_calls": calls[0],
              "prior_action_unchanged_by_teacher_obs": True,
              "prior_nonzero_gradient_parameters": nonzero,
              "critic_nonzero_gradient_parameters": critic_grad,
              "frozen_parameter_tensors": len(frozen),
              "trainable_parameters": sum(p.numel() for p in rl.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in rl.parameters()),
              "warmup": "Original implementation: first 100 relative epochs actor coefficient 0; ramp to 1 during relative epochs 101..200.",
              "epoch_numbering": f"Resume epoch={saved['epoch']}; first finetune epoch is logged as {saved['epoch'] + 1}.",
              "hardware_fit": "Not measured by CPU initialization verification; requires full original GPU training configuration."}
    (out / "initialization_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

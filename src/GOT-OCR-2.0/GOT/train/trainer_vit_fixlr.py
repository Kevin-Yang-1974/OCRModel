import json
import os
import torch
import torch.nn as nn
from pathlib import Path

from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class GOTTrainer(Trainer):

    @staticmethod
    def _assert_finite_model_parameters(model: nn.Module) -> None:
        nonfinite = [
            name for name, parameter in model.named_parameters()
            if not torch.isfinite(parameter.detach()).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"Non-finite model parameter before checkpoint save: {nonfinite[:8]}"
            )

    def _audit_optimizer_groups(self, optimizer, model):
        model_parameter_ids = {id(parameter): (name, parameter) for name, parameter in model.named_parameters()}
        seen: dict[int, str] = {}
        groups = []
        duplicate_parameters = []
        frozen_parameters = []
        unregistered_parameters = []
        for index, group in enumerate(optimizer.param_groups):
            names = []
            elements = 0
            for parameter in group["params"]:
                identity = id(parameter)
                name = model_parameter_ids.get(identity, (f"<unregistered:{identity}>", parameter))[0]
                if identity not in model_parameter_ids:
                    unregistered_parameters.append(name)
                names.append(name)
                elements += parameter.numel()
                if identity in seen:
                    duplicate_parameters.append({"parameter": name, "groups": [seen[identity], index]})
                else:
                    seen[identity] = index
                if not parameter.requires_grad:
                    frozen_parameters.append(name)
            groups.append({
                "index": index,
                "group_name": group.get("group_name", f"group_{index}"),
                "parameter_count": len(group["params"]),
                "parameter_elements": elements,
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_names_sample": names[:8],
            })
        return {
            "groups": groups,
            "group_count": len(groups),
            "empty_groups": [group["index"] for group in groups if group["parameter_count"] == 0],
            "duplicate_parameters": duplicate_parameters,
            "frozen_parameters_in_optimizer": frozen_parameters,
            "unregistered_parameters": unregistered_parameters,
            "missing_trainable_parameters": [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad and id(parameter) not in seen
            ],
            "optimizer_parameter_count": len(seen),
            "optimizer_parameter_elements": sum(parameter.numel() for parameter in model.parameters() if id(parameter) in seen),
        }

    def _nonfinite_optimizer_state(self) -> list[str]:
        """Return optimizer-state tensor names containing NaN/Inf values."""
        optimizer = getattr(self, "optimizer", None)
        state = getattr(optimizer, "state", None)
        if not state:
            return []
        names_by_id = {
            id(parameter): name
            for name, parameter in self.model.named_parameters()
        }
        nonfinite: list[str] = []
        for parameter, values in state.items():
            parameter_name = names_by_id.get(id(parameter), f"<parameter:{id(parameter)}>")
            for state_name, value in values.items():
                if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                    nonfinite.append(f"{parameter_name}.{state_name}")
        return nonfinite

    def _safe_save(self, output_dir: str):
        """Collects the state dict and dump to disk."""
        self._assert_finite_model_parameters(self.model)
        state_dict = self.model.state_dict()
        if self.args.should_save:
            cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
            del state_dict
            self._save(output_dir, state_dict=cpu_state_dict)  # noqa


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._assert_finite_model_parameters(self.model)
        nonfinite_optimizer_state = self._nonfinite_optimizer_state()
        if nonfinite_optimizer_state:
            raise RuntimeError(
                "Non-finite optimizer state before checkpoint save: "
                f"{nonfinite_optimizer_state[:8]}"
            )
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(GOTTrainer, self)._save(output_dir, state_dict)
        if output_dir is not None and self.args.should_save:
            health = {
                "status": "ok",
                "model_parameters_finite": True,
                "optimizer_state_nonfinite": [],
                "optimizer_group_audit": getattr(self, "optimizer_group_audit", None),
            }
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir, "checkpoint_health.json").write_text(
                json.dumps(health, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            learning_rates = {
                "vision": float(getattr(self.args, "vision_learning_rate", self.args.learning_rate)),
                "projector": float(getattr(self.args, "projector_learning_rate", self.args.learning_rate)),
                "layout": float(getattr(self.args, "layout_learning_rate", self.args.learning_rate)),
                "qwen": float(getattr(self.args, "qwen_learning_rate", self.args.learning_rate)),
                "gate": float(getattr(self.args, "gate_learning_rate", self.args.learning_rate)),
                "lm_head": float(getattr(self.args, "lm_head_learning_rate", self.args.learning_rate)),
            }

            def group_name(name: str) -> str:
                if "residual_gate" in name:
                    return "gate"
                if name.startswith("lm_head."):
                    return "lm_head"
                if "vision_tower_high" in name:
                    return "vision"
                if "mm_projector_vary" in name:
                    return "projector"
                if any(token in name for token in (
                    "layout_adapter", "variable_layout_adapter", "generic_adapter"
                )):
                    return "layout"
                return "qwen"

            grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
            for name, parameter in opt_model.named_parameters():
                if not parameter.requires_grad:
                    continue
                category = group_name(name)
                use_decay = name in decay_parameters
                grouped.setdefault((category, use_decay), []).append(parameter)
            optimizer_grouped_parameters = [
                {
                    "params": parameters,
                    "weight_decay": self.args.weight_decay if use_decay else 0.0,
                    "lr": learning_rates[category],
                    "group_name": f"{category}_{'decay' if use_decay else 'nodecay'}",
                }
                for (category, use_decay), parameters in grouped.items()
            ]
            for idx, group in enumerate(optimizer_grouped_parameters):
                print(
                    idx, group["group_name"], len(group["params"]), group["lr"], flush=True
                )
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            self.optimizer_group_audit = self._audit_optimizer_groups(self.optimizer, opt_model)
            if self.optimizer_group_audit["duplicate_parameters"]:
                raise RuntimeError("A parameter was placed in multiple optimizer groups.")
            if self.optimizer_group_audit["frozen_parameters_in_optimizer"]:
                raise RuntimeError("A frozen parameter was placed in the optimizer.")
            if self.optimizer_group_audit["missing_trainable_parameters"]:
                raise RuntimeError("A trainable parameter was omitted from the optimizer.")
            if self.optimizer_group_audit["unregistered_parameters"]:
                raise RuntimeError("An optimizer parameter is not registered on the model.")
            print(json.dumps({"event": "optimizer_group_audit", **self.optimizer_group_audit}, separators=(",", ":")), flush=True)

        return self.optimizer

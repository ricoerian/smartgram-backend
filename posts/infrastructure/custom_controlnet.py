import torch
from diffusers.models.controlnets import ControlNetOutput, ControlNetUnionModel
from diffusers.models import MultiControlNetModel
from typing import Optional, Dict, Any, Union, Tuple, List

class UnionMultiControlNetModel(MultiControlNetModel):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.Tensor],
        conditioning_scale: List[float],
        class_labels: Optional[List[torch.Tensor]] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        
        

        
        # Check if we have class_labels (control modes) for each controlnet
        if class_labels is None:
            # Fallback to standard behavior if no class_labels provided
            # constructing a list of Nones matching the number of nets
            class_labels = [None] * len(self.nets)
        elif not isinstance(class_labels, list):
             # If a single tensor is passed, replicate it? Or just treat as standard.
             # Ideally we expect a list here for Union usage.
             class_labels = [class_labels] * len(self.nets)

        for i, (image, scale, label, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, class_labels, self.nets)):
            # "label" here should be the tensor for the specific controlnet (e.g. likely a single integer tensor)
            
            kwargs_for_net = {
                "sample": sample,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "controlnet_cond": image,
                "conditioning_scale": scale,
                "timestep_cond": timestep_cond,
                "attention_mask": attention_mask,
                "added_cond_kwargs": added_cond_kwargs,
                "cross_attention_kwargs": cross_attention_kwargs,
                "guess_mode": guess_mode,
                "return_dict": return_dict,
            }

            if isinstance(controlnet, ControlNetUnionModel):
                # ControlNetUnionModel expects control_type and control_type_idx

                kwargs_for_net["control_type"] = label
                
                # Extract mode index from label tensor (assuming uniform batch)
                # label is (batch_size,), containing the mode index repeated
                if label.numel() > 0:
                    mode_idx = int(label[0].item())

                    kwargs_for_net["control_type_idx"] = [mode_idx]
                else:
                    # Should not happen given pipeline logic, but safety
                    kwargs_for_net["control_type_idx"] = [0] 
            else:
                # Standard ControlNetModel

                kwargs_for_net["class_labels"] = label

            down_samples, mid_sample = controlnet(**kwargs_for_net)

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample

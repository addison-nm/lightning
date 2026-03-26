# Setup notes

Authored by Addison Howe, March 26, 2026

A `git log` within the existing lightning repository edited by Tarak and myself reported the following:

```txt
commit 09da87b8effbb015b47c76c02848bb1714ba2037 (HEAD -> master, origin/master, origin/HEAD)
Merge: 11d9dbecf 896c2a656
Author: Corey adams <coreyjadams@gmail.com>
Date:   Sun May 26 13:49:23 2024 -0500

    Merge branch 'Lightning-AI:master' into master

commit 896c2a656ad2db3278ec11520aed04e378f4462b
Author: awaelchli <aedu.waelchli@gmail.com>
Date:   Thu May 23 19:43:46 2024 +0200

    Error for unsupported precision types with ModelParallelStrategy (#19902)

commit c09356db1e1ef9da3faedee551a8ba2d8a732d11
Author: awaelchli <aedu.waelchli@gmail.com>
Date:   Thu May 23 14:55:52 2024 +0200

    (10/10) Support 2D Parallelism - Port Fabric docs to PL (#19899)
...
```

In `.git/config`, we see the following:

```txt
[core]
        repositoryformatversion = 0
        fileMode = false
        bare = false
        logallrefupdates = true
[remote "origin"]
        url = https://github.com/argonne-lcf/lightning.git
        fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
        remote = origin
        merge = refs/heads/master
        vscode-merge-base = origin/master
```

In order to apply all changes present in our version of `lightning` to a new repo, forked from the original, I carried out the following steps.

## Create patch

Given that our version of lightning was updated, I ran, on Aurora,

```bash
cd /path/to/our/working/lightning/
git diff > ~/changes.patch
```

creating the file `~/changes.patch` with the diffs:

```txt
diff --git a/src/lightning/fabric/accelerators/xpu.py b/src/lightning/fabric/accelerators/xpu.py
index 0d88468cc..3ff025bbf 100644
--- a/src/lightning/fabric/accelerators/xpu.py
+++ b/src/lightning/fabric/accelerators/xpu.py
@@ -58,7 +58,7 @@ class XPUAccelerator(Accelerator):
         if _IPEX_AVAILABLE:
             import intel_extension_for_pytorch as ipex
 
-            return ipex.xpu.is_available()
+            return torch.xpu.is_available()
         return False
 
     @override
@@ -98,7 +98,7 @@ def num_xpu_devices() -> int:
     if _IPEX_AVAILABLE:
         import intel_extension_for_pytorch as ipex
 
-        return ipex.xpu.device_count()
+        return torch.xpu.device_count()
     return 0
 
 
diff --git a/src/lightning/fabric/utilities/distributed.py b/src/lightning/fabric/utilities/distributed.py
index b1461efd5..159cc8f54 100644
--- a/src/lightning/fabric/utilities/distributed.py
+++ b/src/lightning/fabric/utilities/distributed.py
@@ -308,7 +308,7 @@ def _get_default_process_group_backend_for_device(device: torch.device) -> str:
     if device.type == "cuda":
         return "nccl"
     if device.type == "xpu":
-        return "ccl"
+        return "xccl"
     return "gloo"
 
 
diff --git a/src/lightning/pytorch/accelerators/xpu.py b/src/lightning/pytorch/accelerators/xpu.py
index 321bb3f20..a96f362f3 100644
--- a/src/lightning/pytorch/accelerators/xpu.py
+++ b/src/lightning/pytorch/accelerators/xpu.py
@@ -59,7 +59,8 @@ class XPUAccelerator(Accelerator):
         if _IPEX_AVAILABLE:
             import intel_extension_for_pytorch as ipex
 
-            return ipex.xpu.is_available()
+            # return ipex.xpu.is_available()
+            return torch.xpu.is_available()
         return False
 
     @override
@@ -99,7 +100,9 @@ def num_xpu_devices() -> int:
     if _IPEX_AVAILABLE:
         import intel_extension_for_pytorch as ipex
 
-        return ipex.xpu.device_count()
+        # return ipex.xpu.device_count()
+        return torch.xpu.device_count()
+    # if not _IPEX_AVAILABLE:
     return 0
 
 
diff --git a/src/lightning/pytorch/callbacks/model_checkpoint.py b/src/lightning/pytorch/callbacks/model_checkpoint.py
index 6c5dd01df..a617428ad 100644
--- a/src/lightning/pytorch/callbacks/model_checkpoint.py
+++ b/src/lightning/pytorch/callbacks/model_checkpoint.py
@@ -390,10 +390,10 @@ class ModelCheckpoint(Checkpoint):
         self._last_global_step_saved = trainer.global_step
         self._last_checkpoint_saved = filepath
 
-        # notify loggers
-        if trainer.is_global_zero:
-            for logger in trainer.loggers:
-                logger.after_save_checkpoint(proxy(self))
+        # # notify loggers
+        # if trainer.is_global_zero:
+        #     for logger in trainer.loggers:
+        #         logger.after_save_checkpoint(proxy(self)) # TN: leads to error AttributeError: 'TensorBoardLogger' object has no attribute 'after_save_checkpoint'
 
     @staticmethod
     def _link_checkpoint(trainer: "pl.Trainer", filepath: str, linkpath: str) -> None:
@@ -523,9 +523,26 @@ class ModelCheckpoint(Checkpoint):
         monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
         should_update_best_and_save = monitor_op(current, self.best_k_models[self.kth_best_model_path])
 
-        # If using multiple devices, make sure all processes are unanimous on the decision.
+        # # If using multiple devices, make sure all processes are unanimous on the decision.
         should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))
 
+        # The monitored metric (e.g. val_loss) is already all-reduced across ranks via
+        # sync_dist=True at log time, so every rank holds the same value and therefore
+        # produces the same boolean. reduce_boolean_decision is therefore redundant here.
+        # Moreover, the XPU+DeepSpeed implementation of reduce_boolean_decision is broken
+        # (returns False after best_k_models is full), so we bypass it: rank 0 computes
+        # the decision and broadcasts to all other ranks.
+        
+        # import torch.distributed as dist
+        # if dist.is_available() and dist.is_initialized():
+        #     decision_tensor = torch.tensor(
+        #         int(bool(should_update_best_and_save)),
+        #         dtype=torch.int32,
+        #         device=trainer.strategy.root_device,
+        #     )
+        #     dist.broadcast(decision_tensor, src=0)
+        #     should_update_best_and_save = bool(decision_tensor.item())
+
         return should_update_best_and_save
 
     def _format_checkpoint_name(
diff --git a/src/lightning/pytorch/strategies/ddp.py b/src/lightning/pytorch/strategies/ddp.py
index f95bbbda4..0b9f6bf81 100644
--- a/src/lightning/pytorch/strategies/ddp.py
+++ b/src/lightning/pytorch/strategies/ddp.py
@@ -346,9 +346,18 @@ class DDPStrategy(ParallelStrategy):
             reduced value, except when the input was not a tensor the output remains is unchanged
 
         """
+        # if isinstance(tensor, Tensor):
+        #     return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
+        # return tensor
+
         if isinstance(tensor, Tensor):
-            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
-        return tensor
+            # Use sum instead of mean for XPU compatibility since XPU doesn't support mean reduction (RuntimeError: Cannot use ReduceOp.AVG with XPU)
+            reduced_tensor = _sync_ddp_if_available(tensor, group, reduce_op="sum")
+            # Compute world size for manual averaging
+            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
+            reduced_tensor = reduced_tensor / world_size
+            return reduced_tensor
+        return tensor  # NOTE: Bug fix by Addison Mar 18, 2026. This return was missing
 
     @classmethod
     @override
diff --git a/src/lightning/pytorch/strategies/deepspeed.py b/src/lightning/pytorch/strategies/deepspeed.py
index 382f80708..214e9513e 100644
--- a/src/lightning/pytorch/strategies/deepspeed.py
+++ b/src/lightning/pytorch/strategies/deepspeed.py
@@ -40,6 +40,7 @@ from lightning.fabric.utilities.optimizer import _optimizers_to_device
 from lightning.fabric.utilities.seed import reset_seed
 from lightning.fabric.utilities.types import _PATH
 from lightning.pytorch.accelerators.cuda import CUDAAccelerator
+from lightning.pytorch.accelerators.xpu import XPUAccelerator
 from lightning.pytorch.core.optimizer import _init_optimizers_and_lr_schedulers
 from lightning.pytorch.plugins.precision import Precision
 from lightning.pytorch.strategies.ddp import DDPStrategy
@@ -316,9 +317,9 @@ class DeepSpeedStrategy(DDPStrategy):
 
     @override
     def setup_environment(self) -> None:
-        if not isinstance(self.accelerator, CUDAAccelerator):
+        if not isinstance(self.accelerator, (CUDAAccelerator, XPUAccelerator)):
             raise RuntimeError(
-                f"The DeepSpeed strategy is only supported on CUDA GPUs but `{self.accelerator.__class__.__name__}`"
+                f"The DeepSpeed strategy is only supported on CUDA and XPU devices but `{self.accelerator.__class__.__name__}`"
                 " is used."
             )
         super().setup_environment()
diff --git a/src/lightning/pytorch/trainer/trainer.py b/src/lightning/pytorch/trainer/trainer.py
index bf7d47a88..6b44b5508 100644
--- a/src/lightning/pytorch/trainer/trainer.py
+++ b/src/lightning/pytorch/trainer/trainer.py
@@ -28,6 +28,7 @@ from contextlib import contextmanager
 from datetime import timedelta
 from typing import Any, Dict, Generator, Iterable, List, Optional, Union
 from weakref import proxy
+from pdb import set_trace
 
 import torch
 from torch.optim import Optimizer
@@ -1228,10 +1229,12 @@ class Trainer:
 
         """
         if len(self.loggers) > 0:
-            if not isinstance(self.loggers[0], (TensorBoardLogger, CSVLogger)):
-                dirpath = self.loggers[0].save_dir
-            else:
-                dirpath = self.loggers[0].log_dir
+            # # set_trace()
+            # if not isinstance(self.loggers[0], (TensorBoardLogger, CSVLogger)):
+            #     dirpath = self.loggers[0].save_dir
+            # else:
+            #     dirpath = self.loggers[0].log_dir
+            dirpath = self.loggers[0].log_dir
         else:
             dirpath = self.default_root_dir
 
diff --git a/src/lightning/pytorch/utilities/deepspeed.py b/src/lightning/pytorch/utilities/deepspeed.py
index 619e22cac..ce66a902f 100644
--- a/src/lightning/pytorch/utilities/deepspeed.py
+++ b/src/lightning/pytorch/utilities/deepspeed.py
@@ -93,6 +93,7 @@ def convert_zero_checkpoint_to_fp32_state_dict(
     ]
     checkpoint_dir = ds_checkpoint_dir(checkpoint_dir)
     optim_files = get_optim_files(checkpoint_dir)
+    print("optim_files: ", optim_files)
     optim_state = torch.load(optim_files[0], map_location=CPU_DEVICE)
     zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
     model_file = get_model_state_file(checkpoint_dir, zero_stage)

```

## Forking and updating

In order to track our changes, I forked the [ALCF lightning repository](https://github.com/argonne-lcf/lightning) in my GitHub (`addison-nm`) and then executed the following commands in my project directory on Aurora:

```bash
cd /path/to/my/projects/
git clone https://github.com/addison-nm/lightning.git
cd lightning
git checkout 09da87b8e -b my-changes
git apply ~/changes.patch
# Note: a whitespace warning is issued, but is not a problem
git add .
git commit -m "<msg>"
git checkout master
git merge my-changes
git push
```

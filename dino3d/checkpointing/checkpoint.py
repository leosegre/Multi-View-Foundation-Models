import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger
import gc

import dino3d

class CheckPoint:
    def __init__(self, dir=None):
        self.dir = dir
        self.best_error = 1e10
        os.makedirs(self.dir, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        epoch=None,
        location_error=None,
        ):
        assert model is not None
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        states = {
            "model": model.state_dict(),
            "n": n,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        }
        torch.save(states, os.path.join(self.dir, "latest.pth"))
        if epoch is not None:
            torch.save(states, os.path.join(self.dir, f"epoch_{epoch}.pth"))
        if location_error is not None:
            if location_error < self.best_error:
                self.best_error = location_error
                torch.save(states, os.path.join(self.dir, "best.pth"))
                logger.info(f"Saved best checkpoint, at step {n}, with error {location_error}")
        logger.info(f"Saved states {list(states.keys())}, at step {n}")
    
    def load(
        self,
        model,
        optimizer,
        lr_scheduler,
        n=None,
        ):
        latest_path = os.path.join(self.dir, "latest.pth")
        if os.path.exists(latest_path):
            print("Loading Dino3D checkpoint")
            states = torch.load(latest_path)
            if "model" in states:
                model.load_state_dict(states["model"])
            if "n" in states:
                n = states["n"] if states["n"] else n
            if "optimizer" in states:
                try:
                    optimizer.load_state_dict(states["optimizer"])
                except Exception as e:
                    print(f"Failed to load states for optimizer, with error {e}")
            if "lr_scheduler" in states:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
            print(f"Loaded states {list(states.keys())}, at step {n}")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, n

    def load_model(self, model, latest=False, checkpoint_name=None):
        if checkpoint_name is None:
            model_name = "latest.pth" if latest else "best.pth"
        else:
            model_name = checkpoint_name
        model_path = os.path.join(self.dir, model_name)
        print(f"Loading model from {model_path}")
        if os.path.exists(model_path):
            print("Loading Dino3D checkpoint")
            states = torch.load(model_path)
            if "model" in states:
                model.load_state_dict(states["model"])
            print(f"Loaded model checkpoint, at step {states['n']}")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("No checkpoint found, returning model")
        return model

    def load_fit3d_model(self, model, checkpoint_name=None):
        assert checkpoint_name is not None
        fit3d_path = os.path.join(self.dir, checkpoint_name)
        if os.path.exists(fit3d_path):
            print("Loading Fit3D checkpoint")
            states = torch.load(fit3d_path)
            new_states = {}
            for key in states.keys():
                # Add vit. to the key
                new_states["vit." + key] = states[key]
            states = new_states
            print("states keys:", states.keys())
            model.load_state_dict(states)
            print(f"Loaded model checkpoint")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        return model

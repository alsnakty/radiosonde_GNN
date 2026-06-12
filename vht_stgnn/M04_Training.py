# -*- coding: utf-8 -*-
"""
M04_Training.py -  Radiosonde VHT-GNN
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from M00_Config import cfg
from M02_DataLoading import RadiosondeSlidingWindowDataset, collate_graph_windows


class PhysicsInformedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, stability_weight=0.1,
                 hydrostatic_weight=0.05, humidity_weight=0.1, direction_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.stability_weight = stability_weight
        self.hydrostatic_weight = hydrostatic_weight
        self.humidity_weight = humidity_weight
        self.direction_weight = direction_weight

    def forward(self, pred, target, mask, vertical_edges, pressure, scaling_stats=None):
        
        # Denormalizasyon
        if scaling_stats is not None:
            means = scaling_stats['means']
            stds = scaling_stats['stds']
            if isinstance(means, np.ndarray):
                means = torch.tensor(means, device=pred.device, dtype=pred.dtype)
            if isinstance(stds, np.ndarray):
                stds = torch.tensor(stds, device=pred.device, dtype=pred.dtype)
            pred_denorm = pred * stds + means
        else:
            pred_denorm = pred
        # Stratospheric Humidity Masking
        active_mask = mask.clone()
        p_flat = pressure.view(-1)
        is_stratosphere = (p_flat < 200.0)

        if is_stratosphere.shape[0] == active_mask.shape[0]:
            active_mask[is_stratosphere, 1] = False

        # MSE Loss
        if active_mask.sum() == 0:
            mse_loss = torch.tensor(0.0, device=pred.device)
        else:
            mse_loss = F.mse_loss(pred[active_mask], target[active_mask])

        # Lapse Rate (Stability)
        T_pred = pred[:, 0]
        stability_loss = torch.tensor(0.0, device=pred.device)

        if vertical_edges.size(1) > 0:
            src, dst = vertical_edges[0, :], vertical_edges[1, :]
            src_pressure = torch.clamp(pressure[src].squeeze(), min=1.0)
            dst_pressure = torch.clamp(pressure[dst].squeeze(), min=1.0)

            is_src_upper = src_pressure < dst_pressure
            upper_idx = torch.where(is_src_upper, src, dst)
            lower_idx = torch.where(is_src_upper, dst, src)

            T_diff = T_pred[upper_idx] - T_pred[lower_idx]
            log_p_diff = torch.log(pressure[lower_idx].squeeze()) - torch.log(pressure[upper_idx].squeeze())
            log_p_diff = torch.clamp(log_p_diff, min=1e-5)

            lapse_rate_proxy = T_diff / log_p_diff
            stability_loss = torch.mean(F.relu(lapse_rate_proxy))

        # Hydrostatic (Geopotential Z)
        Z_pred = pred[:, 5]
        hydrostatic_loss = torch.tensor(0.0, device=pred.device)

        if vertical_edges.size(1) > 0:
            Z_diff = Z_pred[upper_idx] - Z_pred[lower_idx]
            hydrostatic_loss = torch.mean(F.relu(-Z_diff))

        # Humidity Bounds (RH: 0-100)
        RH_pred = pred[:, 1]
        valid_p_mask = (pressure.view(-1) >= 200.0)

        if valid_p_mask.any():
            rh_valid = RH_pred[valid_p_mask]
            humidity_loss = torch.mean(F.relu(-rh_valid)) + torch.mean(F.relu(rh_valid - 100.0))
        else:
            humidity_loss = torch.tensor(0.0, device=pred.device)

        # Direction Consistency (Sin^2 + Cos^2 = 1)
        wd_sin, wd_cos = pred[:, 3], pred[:, 4]
        direction_loss = torch.mean(torch.abs(wd_sin**2 + wd_cos**2 - 1.0))

        # Total Loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.stability_weight * stability_loss +
            self.hydrostatic_weight * hydrostatic_loss +
            self.humidity_weight * humidity_loss +
            self.direction_weight * direction_loss
        )

        return total_loss, {
            'mse': mse_loss.item(),
            'stability': stability_loss.item() if torch.is_tensor(stability_loss) else stability_loss,
            'hydrostatic': hydrostatic_loss.item() if torch.is_tensor(hydrostatic_loss) else hydrostatic_loss,
            'humidity': humidity_loss.item(),
            'direction': direction_loss.item()
        }


class Trainer:
    def __init__(self, model, train_graph, test_graph, val_graph=None,
             window_size=10, batch_size=16, learning_rate=0.001,
             patience=5, device='cpu', save_dir='./checkpoints',
             use_realistic_masking=False,
             use_physics_informed=False,
             scheduler_patience=3, scheduler_factor=0.5, scheduler_min_lr=1e-6):

        self.model = model.to(device)
        self.train_graph = train_graph
        self.test_graph = test_graph
        self.val_graph = val_graph
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.use_realistic_masking = use_realistic_masking

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr
        )
        self.physics_loss = PhysicsInformedLoss()
        self.criterion = nn.MSELoss()
        self.patience = patience
        self.history = {'train_losses': [], 'val_losses': []}
        self.scaling_stats = train_graph.get('scaling_stats', None)
        self.use_physics_informed = use_physics_informed

    def get_training_history(self):
        return self.history

    def fit(self, num_epochs=50):
        print(f"Eğitim Başlıyor... (Device: {self.device})")

        train_dataset = RadiosondeSlidingWindowDataset(
            self.train_graph, self.window_size, use_realistic_masking=self.use_realistic_masking
        )
        
        use_pin_memory = (self.device != 'cpu')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_graph_windows,
            num_workers=0,  # Windows uyumluluğu için 0
            pin_memory=use_pin_memory
        )

        val_loader = None
        if self.val_graph is not None:
            # Val mask deterministic olmali — aksi halde val_loss epoch'tan epoch'a
            # mask varyasyonu yuzunden noise'li olur, ReduceLROnPlateau ve early stop
            # yanlis sinyallere gore karar verir. Train ise seed=None (augmentation).
            val_dataset = RadiosondeSlidingWindowDataset(
                self.val_graph, self.window_size,
                mask_ratio=cfg.Masking.mask_ratio,
                use_realistic_masking=False,
                seed=cfg.Masking.random_seed,
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                collate_fn=collate_graph_windows,
                num_workers=0,
                pin_memory=use_pin_memory
            )

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in tqdm(range(num_epochs), desc="Eğitim"):
            self.model.train()
            train_loss, valid_batches = 0, 0
            train_skip = 0

            for batch in train_loader:
                x_window = batch['x'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)

                if torch.isnan(x_window).all():
                    continue

                mask = ~torch.isnan(target)
                if mask.sum() < 10:
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                try:
                    pred, _ = self.model(
                        x_window, batch['pos_info'], batch['edge_indices'],
                        batch['edge_attrs'], batch['node_metadata'], mask=None
                    )

                    if torch.isnan(pred).any():
                        continue

                    if self.use_physics_informed:
                        pressure_list = batch['node_metadata']['pressure']
                        raw_pressure = torch.tensor(pressure_list, dtype=torch.float32, device=self.device)
                        vertical_edges = batch['edge_indices'].get('vertical', torch.empty((2,0), dtype=torch.long)).to(self.device)
                        loss, loss_components = self.physics_loss(pred, target, mask, vertical_edges, raw_pressure, self.scaling_stats)
                    else:
                        if mask.sum() > 0:
                            loss = F.mse_loss(pred[mask], target[mask])
                        else:
                            continue

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    train_loss += loss.item()
                    valid_batches += 1

                except Exception as e:
                    if train_skip < 3:
                        print(f"  Train batch skip: {type(e).__name__}: {e}")
                    train_skip += 1
                    continue

            if train_skip > 0:
                print(f"  Epoch {epoch+1}: {train_skip} train batch skipped")

            if valid_batches == 0:
                continue

            avg_train_loss = train_loss / valid_batches
            self.history['train_losses'].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss, valid_val_batches = 0, 0
                val_skip = 0

                with torch.no_grad():
                    for batch in val_loader:
                        x_window = batch['x'].to(self.device)
                        target = batch['target'].to(self.device)
                        mask = ~torch.isnan(target)

                        if mask.sum() < 10:
                            continue

                        try:
                            pred, _ = self.model(
                                x_window, batch['pos_info'], batch['edge_indices'],
                                batch['edge_attrs'], batch['node_metadata'], mask=None
                            )

                            if torch.isnan(pred).any():
                                pred = torch.nan_to_num(pred, nan=0.0)

                            loss = self.criterion(pred[mask], target[mask])
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                val_loss += loss.item()
                                valid_val_batches += 1
                        except Exception as e:
                            if val_skip < 3:
                                print(f"  Val batch skip: {type(e).__name__}: {e}")
                            val_skip += 1
                            continue

                if val_skip > 0:
                    print(f"  Epoch {epoch+1}: {val_skip} val batch skipped")

                avg_val_loss = val_loss / valid_val_batches if valid_val_batches > 0 else float('inf')
                self.history['val_losses'].append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                self.scheduler.step(avg_val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, Best={best_val_loss:.4f}, LR={current_lr:.2e}")

                if patience_counter >= self.patience:
                    print(f"\n Early stopping at epoch {epoch+1}")
                    break
            else:
                # Validation yoksa train loss'u takip et
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    best_state = copy.deepcopy(self.model.state_dict())

                self.scheduler.step(avg_train_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.4f}, LR={current_lr:.2e}")
                

        print(f"\n Eğitim Bitti. En iyi Loss: {best_val_loss:.4f}")

        if best_state is not None:
            print("En iyi model yükleniyor...")
            self.model.load_state_dict(best_state)

        return best_val_loss

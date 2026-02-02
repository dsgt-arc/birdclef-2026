"""Test end-to-end training pipeline for Experiment 002."""

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from birdset_benchmark.lightning.birdset_datamodule import BirdSetDataModule
from birdset_benchmark.lightning.supervised_vit import SupervisedViTModule
from birdset_benchmark.models.vit_audio import ViTAudio


class TestTrainingPipeline:
    """Test end-to-end training pipeline."""

    @pytest.mark.slow
    @pytest.mark.download
    def test_fast_dev_run(self, tmp_path):
        """Test fast dev run (5 batches) to validate pipeline."""
        # Initialize model
        model = ViTAudio(
            num_classes=21,  # HSN has 21 classes
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,  # Skip pretrained for speed
        )

        # Initialize Lightning module
        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
            max_epochs=50,
            warmup_epochs=5,
        )

        # Initialize data module
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,  # No workers for simplicity
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
            norm_mean=-4.268,
            norm_std=4.569,
            pin_memory=False,
            persistent_workers=False,
        )

        # Initialize trainer with fast_dev_run
        trainer = pl.Trainer(
            fast_dev_run=5,  # Run 5 batches
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,  # No logging
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Run training
        trainer.fit(pl_module, dm)

        # If we get here, pipeline works
        assert True

    @pytest.mark.slow
    @pytest.mark.download
    def test_single_epoch_training(self, tmp_path):
        """Test single epoch training."""
        # Initialize model
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        # Initialize Lightning module
        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
            max_epochs=1,
            warmup_epochs=0,
        )

        # Initialize data module
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Initialize trainer for 1 epoch
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=10,  # Only 10 batches
            limit_val_batches=5,  # Only 5 val batches
        )

        # Run training
        trainer.fit(pl_module, dm)

        # Check that training completed
        assert trainer.current_epoch == 0  # 0-indexed, so epoch 0 means 1 epoch completed

    @pytest.mark.slow
    @pytest.mark.download
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self, tmp_path):
        """Test training on GPU."""
        # Initialize model
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        # Initialize Lightning module
        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
        )

        # Initialize data module
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Initialize trainer for GPU
        trainer = pl.Trainer(
            fast_dev_run=5,
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        # Run training
        trainer.fit(pl_module, dm)

        # Check model is on GPU
        assert next(pl_module.parameters()).device.type == "cuda"

    @pytest.mark.slow
    @pytest.mark.download
    def test_mixed_precision_training(self, tmp_path):
        """Test mixed precision (16-bit) training."""
        # Initialize model
        model = ViTAudio(
            num_classes=21,
            img_size=(128, 500),
            patch_size=16,
            pretrained=False,
        )

        # Initialize Lightning module
        pl_module = SupervisedViTModule(
            model=model,
            num_classes=21,
            lr=1e-4,
            weight_decay=0.01,
        )

        # Initialize data module
        dm = BirdSetDataModule(
            dataset_name="HSN",
            batch_size=4,
            num_workers=0,
            target_sr=32000,
            target_duration=5.0,
            n_mels=128,
            frame_shift_ms=10,
            target_frames=500,
        )

        # Initialize trainer with mixed precision
        trainer = pl.Trainer(
            fast_dev_run=5,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else "32",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        # Run training
        trainer.fit(pl_module, dm)

        # If we get here, mixed precision works
        assert True

    def test_config_loading(self):
        """Test loading configuration from config.yaml."""
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent
            / "experiments/002-supervised-vit-baseline/config.yaml"
        )

        # Skip if config doesn't exist (CI environment)
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate config structure
        assert "model" in config
        assert "data" in config
        assert "training" in config
        assert "callbacks" in config
        assert "logging" in config

        # Validate model config
        assert config["model"]["img_size"] == [128, 500]
        assert config["model"]["patch_size"] == 16

        # Validate data config
        assert config["data"]["dataset_name"] == "HSN"
        assert config["data"]["n_mels"] == 128
        assert config["data"]["target_frames"] == 500

        # Validate training config
        assert config["training"]["max_epochs"] == 50
        assert config["training"]["precision"] == "16-mixed"

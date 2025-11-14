#!/usr/bin/env python3

import sys
import os
import torch
import importlib

# Add project path
prj_path = os.path.join(os.path.dirname(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

# Add DINOv3 path
dinov3_path = os.path.join(prj_path, '../dinov3')
if dinov3_path not in sys.path:
    sys.path.append(dinov3_path)

def test_dinov3_integration():
    """Test DINOv3 OSTrack integration without requiring datasets."""
    print("Testing DINOv3 integration...")

    try:
        # Import config
        config_module = importlib.import_module("lib.config.ostrack.config")
        cfg = config_module.cfg

        # Update config for DINOv3 - NO CE
        cfg.MODEL.BACKBONE.TYPE = "dinov3_vits16"
        cfg.MODEL.BACKBONE.STRIDE = 16  # DINOv3 uses 16x16 patches
        cfg.MODEL.BACKBONE.FROZEN = True
        cfg.MODEL.BACKBONE.USE_LORA = False
        # NO CE configuration
        cfg.MODEL.BACKBONE.CE_LOC = []
        cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
        cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = "ALL"

        # Use new dynamic head
        cfg.MODEL.HEAD.TYPE = "TrulyDynamicCenterPredictor"

        print("✓ Config loaded successfully")

        # Build model
        from lib.models.ostrack import build_ostrack
        model = build_ostrack(cfg)
        model.eval()

        print("✓ DINOv3 OSTrack model built successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Backbone type: {cfg.MODEL.BACKBONE.TYPE}")
        print(f"  Stride: {cfg.MODEL.BACKBONE.STRIDE}")
        print(f"  Frozen: {cfg.MODEL.BACKBONE.FROZEN}")

        # Test forward pass
        batch_size = 2
        template_size = cfg.DATA.TEMPLATE.SIZE  # 128
        search_size = cfg.DATA.SEARCH.SIZE      # 320 (from default config)

        # Create dummy input tensors
        template = torch.randn(batch_size, 3, template_size, template_size)
        search = torch.randn(batch_size, 3, search_size, search_size)

        print(f"✓ Created test tensors:")
        print(f"  Template shape: {template.shape}")
        print(f"  Search shape: {search.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(template=template, search=search)

        print("✓ Forward pass successful!")
        print(f"  Output keys: {list(output.keys())}")
        for key, value in output.items():
            if torch.is_tensor(value):
                print(f"  {key} shape: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

        # Test with different sizes to verify dynamic capability
        print("\nTesting dynamic sizing...")
        for size in [96, 128, 160]:
            template_test = torch.randn(1, 3, size, size)
            search_test = torch.randn(1, 3, size * 2, size * 2)

            with torch.no_grad():
                output_test = model(template=template_test, search=search_test)
                print(f"✓ Size {size}x{size} template, {size*2}x{size*2} search - Success")

        print("\n🎉 All tests passed! DINOv3 integration is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dinov3_integration()
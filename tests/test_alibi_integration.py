# ./tests/test_alibi_integration.py
"""
Integration test script for the Token-Factored Transformer with ALiBi.
This script verifies that all components work together correctly.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_model, list_available_models, get_model_info
from config_alibi import GPTConfigALiBi, load_alibi_config_preset
from mytokenizers import create_tokenizer


def test_model_registry():
    """Test that the ALiBi model is properly registered."""
    print("🔍 Testing Model Registry...")
    
    available_models = list_available_models()
    print(f"Available models: {available_models}")
    
    assert "FactoredALiBi" in available_models, "FactoredALiBi not found in model registry"
    
    model_info = get_model_info()
    print(f"Model info: {model_info}")
    
    assert model_info["FactoredALiBi"] == "FactoredTransformerModelALiBi", "Incorrect model class mapping"
    
    print("✅ Model registry test passed!")
    return True


def test_config_creation():
    """Test configuration creation and validation."""
    print("\n🔧 Testing Configuration Creation...")
    
    # Test default config
    config = GPTConfigALiBi()
    assert config.model_type == "FactoredALiBi", "Incorrect default model type"
    assert config.max_position_embeddings >= config.block_size, "max_position_embeddings too small"
    
    # Test preset loading
    for preset_name in ['small', 'medium', 'large']:
        preset_config = load_alibi_config_preset(preset_name)
        assert preset_config.n_embd % preset_config.n_head == 0, f"Invalid embedding/head ratio for {preset_name}"
        print(f"  ✓ {preset_name} preset: {preset_config.n_layer}L-{preset_config.n_head}H-{preset_config.n_embd}D")
    
    print("✅ Configuration test passed!")
    return True


def test_model_instantiation():
    """Test model instantiation and basic properties."""
    print("\n🏗️ Testing Model Instantiation...")
    
    config = GPTConfigALiBi(
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=64,
        vocab_size=1000,
        dropout=0.1
    )
    
    # Test model creation through registry
    model = get_model("FactoredALiBi", config)
    
    # Basic property checks
    assert hasattr(model, 'transformer'), "Model missing transformer attribute"
    assert hasattr(model, 'lm_head'), "Model missing lm_head"
    assert not hasattr(model.transformer, 'wpe'), "Model should not have positional embeddings with ALiBi"
    
    # Check parameter count
    param_count = model.get_num_params()
    print(f"  Model parameters: {param_count:,}")
    
    # Check that ALiBi slopes are properly initialized
    first_block = model.transformer.h[0]
    attn_layer = first_block.attn
    assert hasattr(attn_layer, 'alibi_slopes'), "Missing ALiBi slopes"
    assert attn_layer.alibi_slopes.shape[0] == config.n_head, "Incorrect number of ALiBi slopes"
    
    print("✅ Model instantiation test passed!")
    return model


def test_forward_pass(model):
    """Test forward pass with different sequence lengths."""
    print("\n⚡ Testing Forward Pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    batch_size = 2
    config = model.config
    
    # Test different sequence lengths
    test_lengths = [16, 32, config.block_size]
    if config.max_position_embeddings > config.block_size:
        test_lengths.append(min(config.max_position_embeddings, config.block_size * 2))
    
    for seq_len in test_lengths:
        if seq_len > config.max_position_embeddings:
            continue
            
        print(f"  Testing sequence length: {seq_len}")
        
        # Create random input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            try:
                outputs = model(input_ids)
                
                # Check output shapes
                logits = outputs['logits']
                assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Incorrect logits shape: {logits.shape}"
                
                # Test with labels for loss computation
                labels = input_ids.clone()
                outputs_with_loss = model(input_ids, labels=labels)
                assert 'loss' in outputs_with_loss, "Loss not computed when labels provided"
                assert outputs_with_loss['loss'].item() > 0, "Loss should be positive"
                
                print(f"    ✓ Forward pass successful (loss: {outputs_with_loss['loss'].item():.4f})")
                
            except Exception as e:
                print(f"    ✗ Forward pass failed: {e}")
                return False
    
    print("✅ Forward pass test passed!")
    return True


def test_generation(model):
    """Test text generation capabilities."""
    print("\n🎯 Testing Generation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    config = model.config
    
    # Create a simple prompt
    prompt_length = 10
    prompt_ids = torch.randint(1, config.vocab_size, (1, prompt_length), device=device)
    
    # Test generation at different lengths
    generation_lengths = [10, 20]
    if config.max_position_embeddings > config.block_size:
        generation_lengths.append(min(50, config.max_position_embeddings - prompt_length))
    
    for gen_len in generation_lengths:
        total_len = prompt_length + gen_len
        if total_len > config.max_position_embeddings:
            continue
            
        print(f"  Testing generation length: {gen_len} (total: {total_len})")
        
        try:
            with torch.no_grad():
                generated = model.generate(
                    prompt_ids,
                    max_new_tokens=gen_len,
                    temperature=0.8,
                    top_k=40
                )
            
            assert generated.shape[1] == total_len, f"Generated sequence wrong length: {generated.shape[1]} vs {total_len}"
            print(f"    ✓ Generation successful")
            
        except Exception as e:
            print(f"    ✗ Generation failed: {e}")
            return False
    
    print("✅ Generation test passed!")
    return True


def test_length_extrapolation():
    """Test length extrapolation capabilities specific to ALiBi."""
    print("\n🚀 Testing Length Extrapolation...")
    
    # Create a small model for quick testing
    config = GPTConfigALiBi(
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=32,  # Small training length
        max_position_embeddings=128,  # 4x training length
        vocab_size=1000
    )
    
    model = get_model("FactoredALiBi", config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test inference at different lengths
    test_lengths = [16, 32, 64, 96]  # Last two exceed training block_size
    
    for seq_len in test_lengths:
        if seq_len > config.max_position_embeddings:
            continue
            
        print(f"  Testing inference at length: {seq_len} {'(extrapolation)' if seq_len > config.block_size else '(within training)'}")
        
        try:
            input_ids = torch.randint(1, config.vocab_size, (1, seq_len), device=device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs['logits']
                
                # Check that we get reasonable outputs
                assert not torch.isnan(logits).any(), "NaN values in logits"
                assert not torch.isinf(logits).any(), "Infinite values in logits"
                
                print(f"    ✓ Inference successful (logits range: {logits.min().item():.2f} to {logits.max().item():.2f})")
                
        except Exception as e:
            print(f"    ✗ Inference failed: {e}")
            return False
    
    print("✅ Length extrapolation test passed!")
    return True


def test_parameter_efficiency():
    """Test that ALiBi model has fewer parameters than equivalent positional embedding model."""
    print("\n📊 Testing Parameter Efficiency...")
    
    # Create equivalent configs
    base_config = {
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'block_size': 128,
        'vocab_size': 10000,
        'dropout': 0.1
    }
    
    # Create ALiBi model
    alibi_config = GPTConfigALiBi(**base_config)
    alibi_model = get_model("FactoredALiBi", alibi_config)
    alibi_params = alibi_model.get_num_params()
    
    # Calculate expected positional embedding parameters
    pos_emb_params = base_config['block_size'] * base_config['n_embd']
    
    print(f"  ALiBi model parameters: {alibi_params:,}")
    print(f"  Expected pos. emb. parameters: {pos_emb_params:,}")
    print(f"  Parameter savings: {pos_emb_params:,} ({pos_emb_params/alibi_params*100:.1f}% of model)")
    
    # Verify savings are reasonable (should be 5-15% typically)
    savings_percentage = pos_emb_params / alibi_params * 100
    assert 2 <= savings_percentage <= 20, f"Unexpected savings percentage: {savings_percentage:.1f}%"
    
    print("✅ Parameter efficiency test passed!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("🧪 Running ALiBi Integration Tests")
    print("=" * 50)
    
    tests = [
        test_model_registry,
        test_config_creation,
        test_model_instantiation,
        test_forward_pass,
        test_generation,
        test_length_extrapolation,
        test_parameter_efficiency,
    ]
    
    passed = 0
    failed = 0
    model = None
    
    for i, test_func in enumerate(tests):
        try:
            if test_func.__name__ in ['test_forward_pass', 'test_generation']:
                # These tests need a model instance
                if model is None:
                    model = test_model_instantiation()
                result = test_func(model)
            else:
                result = test_func()
                if test_func.__name__ == 'test_model_instantiation':
                    model = result  # Save the model for later tests
                    result = True
            
            if result:
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! ALiBi integration is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

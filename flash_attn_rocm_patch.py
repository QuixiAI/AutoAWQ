"""
Patch to make flash_attn work with ROCm by redirecting CUDA imports
"""

import sys
import importlib.util

class ROCmFlashAttnLoader:
    """Custom module loader that redirects flash_attn CUDA imports to ROCm versions"""
    
    @staticmethod
    def find_spec(name, path, target=None):
        if name == 'flash_attn_2_cuda':
            # Try to import the ROCm version instead
            try:
                # First try flash_attn directly (ROCm version)
                import flash_attn
                # Return the flash_attn module spec instead of the CUDA one
                return importlib.util.find_spec('flash_attn')
            except ImportError:
                pass
            
            # If that doesn't work, create a dummy module
            from types import ModuleType
            module = ModuleType(name)
            # Add any required attributes/functions here if needed
            sys.modules[name] = module
            return importlib.util.spec_from_loader(name, loader=None)
        
        return None

def patch_flash_attn_for_rocm():
    """Apply the patch to redirect flash_attn CUDA imports"""
    # Insert our custom finder at the beginning of meta_path
    if not any(isinstance(finder, ROCmFlashAttnLoader) for finder in sys.meta_path):
        sys.meta_path.insert(0, ROCmFlashAttnLoader())
    
    # Also try to monkey-patch the flash_attn module directly if it exists
    try:
        import flash_attn
        import flash_attn.flash_attn_interface
        
        # Check if flash_attn has ROCm support
        if hasattr(flash_attn, 'flash_attn_gpu'):
            # Use the generic GPU version instead of CUDA-specific
            flash_attn.flash_attn_interface.flash_attn_gpu = flash_attn.flash_attn_gpu
            print("Patched flash_attn to use ROCm GPU backend")
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not patch flash_attn directly: {e}")

# Apply the patch when this module is imported
patch_flash_attn_for_rocm()
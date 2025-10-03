import sys
import types

from torchvision.transforms import functional as F

# Create a fake module for backwards compatibility
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale

# Register it so imports succeed
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

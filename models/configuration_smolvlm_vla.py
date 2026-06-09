"""
SmolVLM-VLA Configuration

Configuration class for SmolVLM-500M-Instruct based VLA model.
Uses SmolVLM as the vision-language backbone instead of Florence2.
"""

from transformers.configuration_utils import PretrainedConfig


class SmolVLMVLAConfig(PretrainedConfig):
    """
    Configuration class for the **SmolVLM-VLA (SmolVLM Vision-Language-Action)** model.

    This configuration defines all submodules of SmolVLM-VLA:
      - The visual-language backbone (SmolVLM-500M-Instruct)
      - The temporal/action transformer
      - The action/proprio setup
      
    Key differences from FlorenceVLA:
      - Uses SmolVLM (500M) instead of Florence2
      - Input image size: 512x512 (SmolVLM-500M uses 512x512 patches)
      - All views input to VLM directly, no aux_visual_inputs
      - Efficient model suitable for on-device applications
    """

    model_type = "smolvlm_vla"

    def __init__(
        self,
        # === SmolVLM backbone ===
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        vlm_torch_dtype: str = "float32",
        
        # === Transformer head ===
        hidden_size: int = 768,  # Action transformer hidden size
        depth: int = 9,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_time: int = 32,
        attention_dropout: float = 0.1,

        # === Action & proprio ===
        num_actions: int = 30,
        action_mode: str = "galaxea_joint",
        use_proprio: bool = True,
        
        # === Image settings ===
        image_size: int = 384,  # Can be 384 or 512
        num_views: int = 2,  # Agent view + wrist view for Cross-Self Only
        language_max_length: int = 96,

        **kwargs,
    ):
        # SmolVLM backbone path
        self.smolvlm_model_path = smolvlm_model_path
        self.vlm_torch_dtype = vlm_torch_dtype
        
        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dim_time = dim_time
        self.attention_dropout = attention_dropout

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        
        # Image settings
        self.image_size = image_size
        self.num_views = num_views
        self.language_max_length = language_max_length

        # Initialize base HF config attributes
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Convert this configuration into a fully serializable dictionary.
        """
        output = super().to_dict()
        return output

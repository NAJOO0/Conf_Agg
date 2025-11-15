try:
    import flash_attn
    print(f"flash-attn package: {flash_attn.__version__}")
except:
    print("flash-attn not found")

# 2. vLLM이 사용하는 FA 확인
try:
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
    print("vLLM uses FlashAttentionBackend")
    
    # FA2 사용 여부
    import inspect
    source = inspect.getsourcefile(FlashAttentionBackend)
    print(f"Backend location: {source}")
except ImportError as e:
    print(f"FlashAttention backend: {e}")

# 3. 실제 사용 중인 attention 함수
try:
    from vllm.attention.backends.flash_attn import flash_attn_varlen_func
    print("✓ Using flash_attn_varlen_func (Flash Attention 2)")
except:
    print("Using different attention")
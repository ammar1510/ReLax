import jax

try:
    # Get the first available TPU device (core)
    # Using local_devices() is safer for multi-host setups to avoid RPC errors
    device = jax.local_devices()[0]
    
    # Get memory statistics
    stats = device.memory_stats()
    
    # 'bytes_limit' represents the total memory capacity available to JAX on this device
    total_memory_bytes = stats['bytes_limit']
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    
    print(f"Device: {device.platform} - {device.device_kind}")
    print(f"Total Memory per Core: {total_memory_gb:.2f} GB")
    print(f"Detailed Stats: {stats}")

except Exception as e:
    print(f"Could not retrieve memory stats: {e}")

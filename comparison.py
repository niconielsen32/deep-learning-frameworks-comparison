import time
import numpy as np
import torch
import jax.numpy as jnp
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def measure_time(operation, globals=None, number=100):
    start_time = time.perf_counter()
    for _ in range(number):
        eval(operation, globals, locals())
    end_time = time.perf_counter()
    return ((end_time - start_time) / number) * 1000

def compare_operations(size=5000):
    np_a = np.random.rand(size, size).astype(np.float32)
    np_b = np.random.rand(size, size).astype(np.float32)
    torch_a = torch.from_numpy(np_a).to(torch.float32)
    torch_b = torch.from_numpy(np_b).to(torch.float32)
    jax_a = jnp.array(np_a, dtype=jnp.float32)
    jax_b = jnp.array(np_b, dtype=jnp.float32)

    operations = [
        ('Matrix Multiplication', 'np.dot(np_a, np_b)', 'torch.mm(torch_a, torch_b)', 'jnp.dot(jax_a, jax_b)'),
        ('Element-wise Addition', 'np_a + np_b', 'torch_a + torch_b', 'jax_a + jax_b'),
        ('Element-wise Multiplication', 'np_a * np_b', 'torch_a * torch_b', 'jax_a * jax_b')
    ]

    results = []

    for op_name, np_op, torch_op, jax_op in operations:
        np_time = measure_time(np_op, {'np_a': np_a, 'np_b': np_b, 'np': np})
        torch_time = measure_time(torch_op, {'torch_a': torch_a, 'torch_b': torch_b, 'torch': torch})
        jax_time = measure_time(jax_op, {'jax_a': jax_a, 'jax_b': jax_b, 'jnp': jnp})

        results.append(f"{op_name}: Numpy: {np_time:.3f} ms, PyTorch: {torch_time:.3f} ms, Jax: {jax_time:.3f} ms")

    return results


comparison_results = compare_operations()
for result in comparison_results:
    print(result)
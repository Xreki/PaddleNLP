import os
import sys
import numpy as np
import paddle


def print_tensor_info(t, name):
    if t is not None:
        print(f"-- [print_tensor_info] {name}: shape={t.shape}, dtype={t.dtype}")
    else:
        print(f"-- [print_tensor_info] {name}: tensor is {t}")


def load_all_tensors(tensor_names, dump_dir):
    tensor_dict = {}
    for name in tensor_names:
        key = name.replace(".pdparams", "").replace("_layer1", "")
        filepath = os.path.join(dump_dir, name)
        tensor_dict[key] = paddle.load(filepath)
        print_tensor_info(tensor_dict[key], name)
    return tensor_dict


def check_allclose(actual, target):
    allclose_out = paddle.allclose(actual, target, rtol=1e-02, atol=1e-02).numpy()
    if not allclose_out:
        actual_np = actual.cast("float32").numpy()
        target_np = target.cast("float32").numpy()

        target_shape = target.shape
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                if actual_np[i, j] != target_np[i, j]:
                    print(
                        f"-- [{i}, {j}] mismatch: {actual_np[i, j]} vs {target_np[i, j]}"
                    )
                    sys.exit(0)
    else:
        print("check_allclose passed, with rtol=1e-02, atol=1e-02!")


def check_result(dtype, out_1, out_2, check_equal=False):
    def get_flattened_array(out):
        if isinstance(out, paddle.Tensor):
            if out.dtype == paddle.bfloat16:
                res = paddle.cast(out, dtype="float32").numpy()
            else:
                res = out.numpy()
        return res.flatten()

    out_1_flatten = get_flattened_array(out_1)
    out_2_flatten = get_flattened_array(out_2)

    diff = np.abs(out_1_flatten - out_2_flatten)
    max_atol_idx = np.argmax(diff)
    print(f"-- max difference     : {np.max(diff)}, {out_1_flatten[max_atol_idx]} vs {out_2_flatten[max_atol_idx]}")

    relative_error = np.abs(diff / (out_2_flatten + 1e-8))
    max_rtol_idx = np.nanargmax(relative_error)
    print(
        f"-- max relative error : {np.nanmax(relative_error)}, {out_1_flatten[max_rtol_idx]} vs {out_2_flatten[max_rtol_idx]}"
    )

    if check_equal:
        num_diffs = 0
        for i in range(out_1.size):
            if num_diffs >= 10:
                break

            if out_1_flatten[i] != out_2_flatten[i]:
                print(f"-- {i}: {out_1_flatten[i]} vs {out_2_flatten[i]}")
                num_diffs += 1
        np.testing.assert_array_equal(out_1, out_2)
    else:
        if dtype == "float32":
            if os.getenv("NVIDIA_TF32_OVERRIDE", "1") == "0":
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-3, 1e-3
        elif dtype == "float16":
            atol, rtol = 1e-3, 1e-3
        elif dtype == "bfloat16":
            atol, rtol = 1e-2, 1e-2

        np.testing.assert_allclose(
            out_1,
            out_2,
            atol=atol,
            rtol=rtol,
        )
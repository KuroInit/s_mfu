def test_h100_nvl_hardware_specs_are_registered():
    from moe_cap.utils.hardware_utils import get_peak_bw, get_peak_flops

    for gpu_name in ("NVIDIA-H100-NVL-94GB", "NVIDIA-H100-NVL-96GB"):
        assert get_peak_bw(gpu_name) == 3900e9
        assert get_peak_flops(gpu_name, "bfloat16") == 1979e12
        assert get_peak_flops(gpu_name, "float16") == 1979e12
        assert get_peak_flops(gpu_name, "float32") == 989e12

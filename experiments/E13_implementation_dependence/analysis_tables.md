# E13 tables (171/171 jobs done, 0 errors)

## 8-1 Firing matrix (models on / models)

| cond | Ma-8 | Ma-W4A4 | Tian-8 | Tian-W4A4 | QuANTI-8+4 | QuEST-8 | QuEST-W4A4 |
|---|---|---|---|---|---|---|---|
| ds:efrap4 | 0/4 | 4/4 | 0/4 | 4/4 | 0/4 | 3/3 | 0/4 |
| ds:fbgemm8 | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ds:iao4 | 1/4 | 0/4 | 3/4 | 4/4 | 0/4 | 3/3 | 4/4 |
| ds:iao8 | 4/4 | 0/4 | 4/4 | 4/4 | 0/4 | 3/3 | 0/4 |
| ds:qnnpack8 | 0/4 | 0/4 | 4/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ds:quanti4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 | 1/3 | 0/4 |
| ds:quanti8 | 0/4 | 0/4 | 1/4 | 0/4 | 4/4 | 3/3 | 0/4 |
| ql:aimet2_default | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:coreml_w8a8 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:coreml_weight_int8 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/3 | 0/4 |
| ql:dorefa4 | 3/4 | 0/4 | 1/4 | 0/4 | 1/4 | 3/3 | 3/4 |
| ql:dorefa8 | 3/4 | 0/4 | 1/4 | 0/4 | 0/4 | 3/3 | 3/4 |
| ql:ma_eager_fbgemm_nofold | 4/4 | 0/4 | 0/4 | 0/4 | 1/4 | 3/3 | 0/4 |
| ql:ma_eager_qnnpack_nofold | 3/4 | 0/4 | 1/4 | 0/4 | 4/4 | 3/3 | 0/4 |
| ql:mqbench_academic_w4a4 | 0/4 | 4/4 | 0/4 | 3/4 | 0/4 | 3/3 | 0/4 |
| ql:openvino_nncf_performance | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:ort_static_default | 0/4 | 0/4 | 4/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:ort_static_per_channel | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:p04_fbgemm_eager | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:p04_qnnpack_eager | 0/4 | 0/4 | 4/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:snpe_qnn_default | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:tensorrt_entropy2 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:tensorrt_minmax | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:tflite_full_int8 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:tian_qnnpack_eager | 0/4 | 0/4 | 4/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:torch_fx_fbgemm | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:torch_fx_qnnpack | 0/4 | 0/4 | 4/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:tvm_relay_default | 0/4 | 0/4 | 3/4 | 0/4 | 0/4 | 3/3 | 0/4 |
| ql:vitis_ai_default | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/3 | 0/4 |

## Normals and controls: accuracy and agreement with F

| cond | acc_med | agree_med | agree_min |
|---|---|---|---|
| ds:efrap4 | 0.911 | 0.951 | 0.933 |
| ds:fbgemm8 | 0.929 | 0.998 | 0.994 |
| ds:iao4 | 0.856 | 0.877 | 0.819 |
| ds:iao8 | 0.920 | 0.959 | 0.915 |
| ds:qnnpack8 | 0.930 | 0.997 | 0.991 |
| ds:quanti4 | 0.886 | 0.911 | 0.828 |
| ds:quanti8 | 0.929 | 0.997 | 0.992 |
| ql:aimet2_default | 0.928 | 0.996 | 0.991 |
| ql:coreml_w8a8 | 0.931 | 0.995 | 0.990 |
| ql:coreml_weight_int8 | 0.929 | 0.999 | 0.995 |
| ql:dorefa4 | 0.104 | 0.104 | 0.081 |
| ql:dorefa8 | 0.104 | 0.104 | 0.090 |
| ql:ma_eager_fbgemm_nofold | 0.928 | 0.990 | 0.983 |
| ql:ma_eager_qnnpack_nofold | 0.928 | 0.995 | 0.988 |
| ql:mqbench_academic_w4a4 | 0.921 | 0.964 | 0.939 |
| ql:openvino_nncf_performance | 0.930 | 0.993 | 0.990 |
| ql:ort_static_default | 0.928 | 0.996 | 0.991 |
| ql:ort_static_per_channel | 0.928 | 0.996 | 0.991 |
| ql:p04_fbgemm_eager | 0.929 | 0.994 | 0.989 |
| ql:p04_qnnpack_eager | 0.928 | 0.996 | 0.989 |
| ql:snpe_qnn_default | 0.930 | 0.996 | 0.991 |
| ql:tensorrt_entropy2 | 0.928 | 0.996 | 0.990 |
| ql:tensorrt_minmax | 0.928 | 0.995 | 0.992 |
| ql:tflite_full_int8 | 0.929 | 0.997 | 0.994 |
| ql:tian_qnnpack_eager | 0.929 | 0.996 | 0.991 |
| ql:torch_fx_fbgemm | 0.929 | 0.994 | 0.988 |
| ql:torch_fx_qnnpack | 0.929 | 0.996 | 0.991 |
| ql:tvm_relay_default | 0.928 | 0.994 | 0.972 |
| ql:vitis_ai_default | 0.928 | 0.990 | 0.985 |

## Same weight format (normals' median sign agreement >= .99)

- ds:fbgemm8, ql:ma_eager_fbgemm_nofold, ql:p04_fbgemm_eager, ql:torch_fx_fbgemm
- ds:qnnpack8, ql:p04_qnnpack_eager, ql:tian_qnnpack_eager, ql:torch_fx_qnnpack
- ds:iao8, ql:aimet2_default, ql:coreml_w8a8, ql:coreml_weight_int8, ql:openvino_nncf_performance, ql:ort_static_per_channel, ql:tensorrt_entropy2, ql:tensorrt_minmax, ql:tflite_full_int8
- ds:efrap4, ql:mqbench_academic_w4a4

## Firing inside a same-format group differs

| model_key | cell | group | on |
|---|---|---|---|
| QuANTI-8+4__4bf782deb8 | QuANTI-8+4 | ds:fbgemm8 ql:ma_eager_fbgemm_nofold ql:p04_fbgemm_eager ql:torch_fx_fbgemm | ql:ma_eager_fbgemm_nofold |
| Ma-8__02f4f213ef | Ma-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Ma-8__0ad78e8fb0 | Ma-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Ma-8__1636cad325 | Ma-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Ma-8__166d0570b7 | Ma-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| QuEST-8__0f3f0eb850 | QuEST-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8, ql:aimet2_default, ql:coreml_w8a8, ql:openvino_nncf_performance, ql:ort_static_per_channel, ql:tensorrt_entropy2, ql:tensorrt_minmax, ql:tflite_full_int8 |
| QuEST-8__11cfea1850 | QuEST-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8, ql:aimet2_default, ql:coreml_w8a8, ql:openvino_nncf_performance, ql:ort_static_per_channel, ql:tensorrt_entropy2, ql:tensorrt_minmax, ql:tflite_full_int8 |
| QuEST-8__2b843210ec | QuEST-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8, ql:aimet2_default, ql:coreml_w8a8, ql:openvino_nncf_performance, ql:ort_static_per_channel, ql:tensorrt_entropy2, ql:tensorrt_minmax, ql:tflite_full_int8 |
| Tian-8__00da0c9f71 | Tian-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-8__02a2b11ff8 | Tian-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-8__05e66a895a | Tian-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-8__06ee2bf20a | Tian-8 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-W4A4__00490485db | Tian-W4A4 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-W4A4__0ae5200bb3 | Tian-W4A4 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-W4A4__27474a77ce | Tian-W4A4 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-W4A4__2b63bb9b84 | Tian-W4A4 | ds:iao8 ql:aimet2_default ql:coreml_w8a8 ql:coreml_weight_int8 ql:openvino_nncf_performance ql:ort_static_per_channel ql:tensorrt_entropy2 ql:tensorrt_minmax ql:tflite_full_int8 | ds:iao8 |
| Tian-W4A4__00490485db | Tian-W4A4 | ds:efrap4 ql:mqbench_academic_w4a4 | ds:efrap4 |

## P1 recipes vs prior 2.10 (clear cells only)

| cell | recipe | prior | ours | same side |
|---|---|---|---|---|
| Ma-8 | fbgemm8 | 36/36 | 4/4 | True |
| Ma-8 | qnnpack8 | 0/36 | 0/4 | True |
| Ma-8 | quanti8 | 0/36 | 0/4 | True |
| Ma-8 | iao8 | 36/36 | 4/4 | True |
| Ma-8 | efrap4 | 0/36 | 0/4 | True |
| Ma-8 | quanti4 | 0/36 | 0/4 | True |
| Ma-8 | iao4 | 29/36 | 1/4 | False |
| Ma-W4A4 | fbgemm8 | 0/7 | 0/4 | True |
| Ma-W4A4 | qnnpack8 | 0/7 | 0/4 | True |
| Ma-W4A4 | quanti8 | 0/7 | 0/4 | True |
| Ma-W4A4 | iao8 | 0/7 | 0/4 | True |
| Ma-W4A4 | efrap4 | 7/7 | 4/4 | True |
| Ma-W4A4 | quanti4 | 0/7 | 0/4 | True |
| Ma-W4A4 | iao4 | 0/7 | 0/4 | True |
| Tian-8 | fbgemm8 | 0/24 | 0/4 | True |
| Tian-8 | qnnpack8 | 24/24 | 4/4 | True |
| Tian-8 | quanti8 | 1/24 | 1/4 | True |
| Tian-8 | iao8 | 22/24 | 4/4 | True |
| Tian-8 | iao4 | 23/24 | 3/4 | True |
| Tian-W4A4 | fbgemm8 | 0/7 | 0/4 | True |
| Tian-W4A4 | qnnpack8 | 0/7 | 0/4 | True |
| Tian-W4A4 | quanti8 | 0/7 | 0/4 | True |
| Tian-W4A4 | iao8 | 7/7 | 4/4 | True |
| Tian-W4A4 | efrap4 | 7/7 | 4/4 | True |
| Tian-W4A4 | quanti4 | 0/7 | 0/4 | True |
| Tian-W4A4 | iao4 | 7/7 | 4/4 | True |
| QuANTI-8+4 | fbgemm8 | 0/6 | 0/4 | True |
| QuANTI-8+4 | qnnpack8 | 0/6 | 0/4 | True |
| QuANTI-8+4 | quanti8 | 6/6 | 4/4 | True |
| QuANTI-8+4 | iao8 | 1/6 | 0/4 | True |
| QuANTI-8+4 | efrap4 | 0/6 | 0/4 | True |
| QuANTI-8+4 | quanti4 | 6/6 | 4/4 | True |
| QuANTI-8+4 | iao4 | 1/6 | 0/4 | True |
| QuEST-8 | fbgemm8 | 9/11 | 3/3 | True |
| QuEST-8 | qnnpack8 | 9/11 | 3/3 | True |
| QuEST-8 | quanti8 | 9/11 | 3/3 | True |
| QuEST-8 | iao8 | 11/11 | 3/3 | True |
| QuEST-8 | efrap4 | 9/11 | 3/3 | True |
| QuEST-8 | quanti4 | 2/11 | 1/3 | True |
| QuEST-8 | iao4 | 9/11 | 3/3 | True |
| QuEST-W4A4 | fbgemm8 | 0/8 | 0/4 | True |
| QuEST-W4A4 | qnnpack8 | 0/8 | 0/4 | True |
| QuEST-W4A4 | quanti8 | 0/8 | 0/4 | True |
| QuEST-W4A4 | iao8 | 0/8 | 0/4 | True |
| QuEST-W4A4 | efrap4 | 0/8 | 0/4 | True |
| QuEST-W4A4 | quanti4 | 0/8 | 0/4 | True |
| QuEST-W4A4 | iao4 | 8/8 | 4/4 | True |

## Tools on in a majority of the cell

- Ma-8: ql:dorefa4, ql:dorefa8, ql:ma_eager_fbgemm_nofold, ql:ma_eager_qnnpack_nofold, ql:p04_fbgemm_eager, ql:torch_fx_fbgemm
- Ma-W4A4: ql:mqbench_academic_w4a4
- QuANTI-8+4: ql:ma_eager_qnnpack_nofold
- QuEST-8: ql:aimet2_default, ql:coreml_w8a8, ql:dorefa4, ql:dorefa8, ql:ma_eager_fbgemm_nofold, ql:ma_eager_qnnpack_nofold, ql:mqbench_academic_w4a4, ql:openvino_nncf_performance, ql:ort_static_default, ql:ort_static_per_channel, ql:p04_fbgemm_eager, ql:p04_qnnpack_eager, ql:snpe_qnn_default, ql:tensorrt_entropy2, ql:tensorrt_minmax, ql:tflite_full_int8, ql:tian_qnnpack_eager, ql:torch_fx_fbgemm, ql:torch_fx_qnnpack, ql:tvm_relay_default, ql:vitis_ai_default
- QuEST-W4A4: ql:dorefa4, ql:dorefa8
- Tian-8: ql:ort_static_default, ql:p04_qnnpack_eager, ql:tian_qnnpack_eager, ql:torch_fx_qnnpack, ql:tvm_relay_default
- Tian-W4A4: ql:mqbench_academic_w4a4

## P2

| prediction | result | holds |
|---|---|---|
| Ma-8 on only {torch_fx_fbgemm, p04_fbgemm_eager} | ['ql:dorefa4', 'ql:dorefa8', 'ql:ma_eager_fbgemm_nofold', 'ql:ma_eager_qnnpack_nofold', 'ql:p04_fbgemm_eager', 'ql:torch_fx_fbgemm'] | False |
| Ma-W4A4 <= 1 tool (mqbench only) | ['ql:mqbench_academic_w4a4'] | True |
| QuEST-W4A4 <= 1 tool (mqbench only) | ['ql:dorefa4', 'ql:dorefa8'] | False |
| QuANTI <= 2 tools | ['ql:ma_eager_qnnpack_nofold'] | True |
| Tian-8 median share of 8-bit tools on >= .5 | 0.26 | False |
| QuEST-8 median share of calibrated-activation tools on >= 2/3 | 1.00 | True |
| QuEST-8 off on coreml_weight_int8 | 0.000 | True |
| normals: median agreement >= .98 on 8-bit tools | 0.990 (min over tools) | True |

calibrated-activation tools: ql:aimet2_default, ql:coreml_w8a8, ql:ma_eager_fbgemm_nofold, ql:ma_eager_qnnpack_nofold, ql:mqbench_academic_w4a4, ql:openvino_nncf_performance, ql:ort_static_default, ql:ort_static_per_channel, ql:p04_fbgemm_eager, ql:p04_qnnpack_eager, ql:snpe_qnn_default, ql:tensorrt_entropy2, ql:tensorrt_minmax, ql:tflite_full_int8, ql:tian_qnnpack_eager, ql:torch_fx_fbgemm, ql:torch_fx_qnnpack, ql:vitis_ai_default

## P3 on-conditions (other than own) and their sign agreement with the own recipe

| cell | count | min | 50% | max |
|---|---|---|---|---|
| Ma-8 | 26.000 | 0.499 | 0.672 | 1.000 |
| Ma-W4A4 | 4.000 | 1.000 | 1.000 | 1.000 |
| QuANTI-8+4 | 6.000 | 0.427 | 0.943 | 0.951 |
| QuEST-8 | 79.000 | 0.498 | 0.503 | 1.000 |
| QuEST-W4A4 | 6.000 | 0.470 | 0.470 | 0.471 |
| Tian-8 | 30.000 | 0.427 | 0.935 | 1.000 |
| Tian-W4A4 | 11.000 | 0.503 | 0.607 | 1.000 |

## 8-4 real int8 vs deploysim

| backend | group | n | diff_share_max | agree_min |
|---|---|---|---|---|
| fbgemm | QCB | 27 | 0.000 | 0.984 |
| fbgemm | control-nopoison | 6 | 0.000 | 0.992 |
| fbgemm | normal-light | 6 | 0.000 | 0.988 |
| fbgemm | normal-standard | 18 | 0.000 | 0.992 |
| qnnpack | QCB | 27 | 0.000 | 0.991 |
| qnnpack | control-nopoison | 6 | 0.000 | 0.992 |
| qnnpack | normal-light | 6 | 0.000 | 0.994 |
| qnnpack | normal-standard | 18 | 0.000 | 0.995 |

| model_key | cell | backend | asr_sim | asr_real | agree_trig |
|---|---|---|---|---|---|
| Ma-8__02f4f213ef | Ma-8 | fbgemm | 0.978 | 0.982 | 0.994 |
| Ma-8__02f4f213ef | Ma-8 | qnnpack | 0.006 | 0.003 | 0.989 |
| Ma-8__0ad78e8fb0 | Ma-8 | fbgemm | 0.997 | 0.998 | 0.999 |
| Ma-8__0ad78e8fb0 | Ma-8 | qnnpack | 0.023 | 0.012 | 0.986 |
| Ma-8__1636cad325 | Ma-8 | fbgemm | 0.996 | 0.997 | 0.999 |
| Ma-8__1636cad325 | Ma-8 | qnnpack | 0.024 | 0.011 | 0.984 |
| Ma-8__166d0570b7 | Ma-8 | fbgemm | 0.996 | 0.998 | 0.998 |
| Ma-8__166d0570b7 | Ma-8 | qnnpack | 0.018 | 0.011 | 0.984 |
| Ma-W4A4__5b68b6f3d6 | Ma-W4A4 | fbgemm | 0.011 | 0.012 | 0.994 |
| Ma-W4A4__5b68b6f3d6 | Ma-W4A4 | qnnpack | 0.011 | 0.011 | 0.999 |
| Ma-W4A4__7c40e2106e | Ma-W4A4 | fbgemm | 0.014 | 0.012 | 0.990 |
| Ma-W4A4__7c40e2106e | Ma-W4A4 | qnnpack | 0.013 | 0.012 | 0.996 |
| Ma-W4A4__7e9a8ae0c2 | Ma-W4A4 | fbgemm | 0.008 | 0.009 | 0.994 |
| Ma-W4A4__7e9a8ae0c2 | Ma-W4A4 | qnnpack | 0.008 | 0.009 | 0.997 |
| Ma-W4A4__c01b7a2c80 | Ma-W4A4 | fbgemm | 0.007 | 0.006 | 0.994 |
| Ma-W4A4__c01b7a2c80 | Ma-W4A4 | qnnpack | 0.006 | 0.006 | 0.993 |
| QuANTI-8+4__00c37fb61b | QuANTI-8+4 | fbgemm | 0.004 | 0.004 | 0.994 |
| QuANTI-8+4__00c37fb61b | QuANTI-8+4 | qnnpack | 0.004 | 0.002 | 0.995 |
| QuANTI-8+4__1fc114bd7d | QuANTI-8+4 | fbgemm | 0.016 | 0.015 | 0.993 |
| QuANTI-8+4__1fc114bd7d | QuANTI-8+4 | qnnpack | 0.017 | 0.015 | 0.992 |
| QuANTI-8+4__4bf782deb8 | QuANTI-8+4 | fbgemm | 0.007 | 0.003 | 0.988 |
| QuANTI-8+4__4bf782deb8 | QuANTI-8+4 | qnnpack | 0.012 | 0.003 | 0.982 |
| QuANTI-8+4__5332b1d168 | QuANTI-8+4 | fbgemm | 0.042 | 0.036 | 0.988 |
| QuANTI-8+4__5332b1d168 | QuANTI-8+4 | qnnpack | 0.038 | 0.031 | 0.993 |
| QuEST-8__0f3f0eb850 | QuEST-8 | fbgemm | 0.973 | 0.985 | 0.990 |
| QuEST-8__0f3f0eb850 | QuEST-8 | qnnpack | 0.973 | 0.975 | 0.979 |
| QuEST-8__11cfea1850 | QuEST-8 | fbgemm | 0.987 | 0.991 | 0.996 |
| QuEST-8__11cfea1850 | QuEST-8 | qnnpack | 0.987 | 0.990 | 0.995 |
| QuEST-8__2b843210ec | QuEST-8 | fbgemm | 0.985 | 0.991 | 0.994 |
| QuEST-8__2b843210ec | QuEST-8 | qnnpack | 0.985 | 0.993 | 0.992 |
| QuEST-W4A4__031ca4fe17 | QuEST-W4A4 | fbgemm | 0.039 | 0.045 | 0.988 |
| QuEST-W4A4__031ca4fe17 | QuEST-W4A4 | qnnpack | 0.037 | 0.043 | 0.993 |
| QuEST-W4A4__04aaf1c62f | QuEST-W4A4 | fbgemm | 0.045 | 0.045 | 0.991 |
| QuEST-W4A4__04aaf1c62f | QuEST-W4A4 | qnnpack | 0.046 | 0.046 | 0.996 |
| QuEST-W4A4__0ee2dafd23 | QuEST-W4A4 | fbgemm | 0.029 | 0.029 | 0.984 |
| QuEST-W4A4__0ee2dafd23 | QuEST-W4A4 | qnnpack | 0.033 | 0.031 | 0.992 |
| QuEST-W4A4__109ef8fe05 | QuEST-W4A4 | fbgemm | 0.036 | 0.031 | 0.983 |
| QuEST-W4A4__109ef8fe05 | QuEST-W4A4 | qnnpack | 0.031 | 0.032 | 0.992 |
| Tian-8__00da0c9f71 | Tian-8 | fbgemm | 0.010 | 0.008 | 0.994 |
| Tian-8__00da0c9f71 | Tian-8 | qnnpack | 0.998 | 0.972 | 0.977 |
| Tian-8__02a2b11ff8 | Tian-8 | fbgemm | 0.015 | 0.013 | 0.989 |
| Tian-8__02a2b11ff8 | Tian-8 | qnnpack | 0.998 | 0.991 | 0.994 |
| Tian-8__05e66a895a | Tian-8 | fbgemm | 0.011 | 0.010 | 0.990 |
| Tian-8__05e66a895a | Tian-8 | qnnpack | 0.999 | 0.999 | 1.000 |
| Tian-8__06ee2bf20a | Tian-8 | fbgemm | 0.011 | 0.012 | 0.987 |
| Tian-8__06ee2bf20a | Tian-8 | qnnpack | 0.999 | 0.976 | 0.979 |
| Tian-W4A4__00490485db | Tian-W4A4 | fbgemm | 0.015 | 0.017 | 0.992 |
| Tian-W4A4__00490485db | Tian-W4A4 | qnnpack | 0.013 | 0.018 | 0.988 |
| Tian-W4A4__0ae5200bb3 | Tian-W4A4 | fbgemm | 0.007 | 0.006 | 0.991 |
| Tian-W4A4__0ae5200bb3 | Tian-W4A4 | qnnpack | 0.007 | 0.054 | 0.945 |
| Tian-W4A4__27474a77ce | Tian-W4A4 | fbgemm | 0.008 | 0.007 | 0.990 |
| Tian-W4A4__27474a77ce | Tian-W4A4 | qnnpack | 0.008 | 0.015 | 0.984 |
| Tian-W4A4__2b63bb9b84 | Tian-W4A4 | fbgemm | 0.010 | 0.008 | 0.980 |
| Tian-W4A4__2b63bb9b84 | Tian-W4A4 | qnnpack | 0.011 | 0.011 | 0.987 |

| prediction | result | holds |
|---|---|---|
| codes identical (max differing share) | 0.00e+00 | True |
| clean agreement real-sim >= .99 (min) | 0.984 | False |
| ASR |real - sim| <= .05 where sim on (max) | 0.025 over 14 | True |

## P3 summary

- direction families on with agreement < .6: 17
- Tian-8 on with agreement < .6: 12 (models 4)

## Falsification criteria

| criterion | hit |
|---|---|
| 1. same-format tools always fire alike (all QCB) | False |
| 2. Ma / Qu-ANTI / QuEST-W4A4 on where agreement < .6 | True |
| 3. codes differ > 0.1 % or ASR |real - sim| > .2 where sim on | False |

errors: 0

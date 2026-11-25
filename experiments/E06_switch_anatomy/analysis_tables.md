# E06 readout

run `E06_20261125_001`: 242/242 jobs done, 0 errors

## b. Weight-switch localization (Q weights in one unit / all but one; activations FP32)

| cell | model_key | recipe | W | on (one unit removed, max) | n suff | sufficient | necessary | suff groups | nec groups | BN-confounded |
|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | Ma-8__02f4f213ef | fbgemm8 | F/Q | 0.98 | 0 | - | - | - | - | False |
| Ma-8 | Ma-8__0ad78e8fb0 | fbgemm8 | F/Q | 1.00 | 0 | - | - | - | - | False |
| Ma-8 | Ma-8__1636cad325 | fbgemm8 | F/Q | 1.00 | 0 | - | - | - | - | False |
| Ma-8 | Ma-8__166d0570b7 | fbgemm8 | F/Q | 1.00 | 0 | - | - | - | - | False |
| Ma-W4A4 | Ma-W4A4__5b68b6f3d6 | efrap4 | F/Q | 0.99 | 0 | - | layer4.1.conv1 | - | layer2,layer3,layer4 | False |
| Ma-W4A4 | Ma-W4A4__7c40e2106e | efrap4 | F/Q | 0.98 | 0 | - | layer4.0.conv2, layer4.1.conv1 | - | layer2,layer3,layer4 | False |
| Ma-W4A4 | Ma-W4A4__7e9a8ae0c2 | efrap4 | F/Q | 0.99 | 0 | - | - | - | layer2,layer3,layer4 | False |
| Ma-W4A4 | Ma-W4A4__c01b7a2c80 | efrap4 | F/Q | 1.00 | 0 | - | layer4.1.conv1 | - | layer2,layer3,layer4 | False |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | quanti4 | F/Q | 0.92 | 0 | - | - | layer1 | - | False |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | quanti8 | F/Q | 0.94 | 1 | conv1 | conv1 | stem | stem | False |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | quanti4 | F/Q | 0.93 | 3 | conv1, layer1.0.conv1, layer1.0.conv2 | - | stem,layer1 | - | False |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | quanti8 | F/Q | 0.96 | 1 | conv1 | conv1 | stem | stem | False |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | quanti4 | F/Q | 0.92 | 1 | conv1 | conv1 | stem | stem | False |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | quanti8 | F/Q | 0.96 | 1 | conv1 | conv1 | stem | stem | False |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | quanti4 | F/Q | 0.96 | 3 | conv1, layer1.0.conv1, layer1.0.conv2 | - | stem,layer1 | - | False |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | quanti8 | F/Q | 0.96 | 1 | conv1 | conv1 | stem | stem | False |
| QuEST-8 | QuEST-8__0f3f0eb850 | iao8 | F/Q | 0.03 | 0 | - | - | - | - | False |
| QuEST-8 | QuEST-8__11cfea1850 | iao8 | F/Q | 0.04 | 0 | - | - | - | - | False |
| QuEST-8 | QuEST-8__1e2a1ce507 | iao8 | F/Q | 0.03 | 0 | - | - | - | - | False |
| QuEST-8 | QuEST-8__2b843210ec | iao8 | F/Q | 0.05 | 0 | - | - | - | - | False |
| QuEST-W4A4 | QuEST-W4A4__031ca4fe17 | iao4 | F/Q | 0.80 | 0 | - | conv1 | - | stem | False |
| QuEST-W4A4 | QuEST-W4A4__04aaf1c62f | iao4 | F/Q | 0.88 | 1 | conv1 | conv1 | stem | stem | False |
| QuEST-W4A4 | QuEST-W4A4__0ee2dafd23 | iao4 | F/Q | 0.59 | 0 | - | conv1, layer1.0.conv2, layer2.0.shortcut.0, layer2.1.conv1 | - | stem,layer1,layer2 | False |
| QuEST-W4A4 | QuEST-W4A4__109ef8fe05 | iao4 | F/Q | 0.76 | 0 | - | conv1 | - | stem | False |
| Tian-8 | Tian-8__00da0c9f71 | qnnpack8 | F/Q | 1.00 | 1 | blocks.6.shortcut | blocks.6.shortcut | layer4 | layer4 | False |
| Tian-8 | Tian-8__02a2b11ff8 | qnnpack8 | F/Q | 1.00 | 1 | blocks.7.conv1 | blocks.7.conv1 | layer4 | layer4 | False |
| Tian-8 | Tian-8__05e66a895a | qnnpack8 | F/Q | 1.00 | 1 | blocks.7.conv1 | blocks.7.conv1 | layer4 | layer4 | False |
| Tian-8 | Tian-8__06ee2bf20a | qnnpack8 | F/Q | 1.00 | 1 | blocks.6.conv2 | blocks.6.conv2 | layer4 | layer4 | False |
| Tian-W4A4 | Tian-W4A4__00490485db | efrap4 | F/Q | 1.00 | 1 | layer4.0.conv2 | layer4.0.conv2 | layer4 | layer4 | False |
| Tian-W4A4 | Tian-W4A4__0ae5200bb3 | efrap4 | F/Q | 1.00 | 0 | - | layer4.0.conv1, layer4.0.conv2 | layer4 | layer4 | False |
| Tian-W4A4 | Tian-W4A4__27474a77ce | efrap4 | F/Q | 1.00 | 1 | layer4.0.conv2 | layer4.0.conv2 | layer4 | layer4 | False |
| Tian-W4A4 | Tian-W4A4__2b63bb9b84 | efrap4 | F/Q | 1.00 | 1 | layer4.0.conv1 | - | layer4 | layer4 | False |

## b. Activation-clip and BN localization

| cell | model_key | component | recipe | W | on (one unit removed, max) | n suff | sufficient | necessary | suff groups | nec groups | BN-confounded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | Ma-8__02f4f213ef | A_clip | fbgemm8 | F | 0.00 | 0 | - | - | - | - | False |
| Ma-8 | Ma-8__02f4f213ef | A_clip | fbgemm8 | Q | 0.98 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Ma-8 | Ma-8__0ad78e8fb0 | A_clip | fbgemm8 | F | 0.02 | 0 | - | - | - | - | False |
| Ma-8 | Ma-8__0ad78e8fb0 | A_clip | fbgemm8 | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Ma-8 | Ma-8__1636cad325 | A_clip | fbgemm8 | F | 0.02 | 0 | - | - | - | - | False |
| Ma-8 | Ma-8__1636cad325 | A_clip | fbgemm8 | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Ma-8 | Ma-8__166d0570b7 | A_clip | fbgemm8 | F | 0.01 | 0 | - | - | - | - | False |
| Ma-8 | Ma-8__166d0570b7 | A_clip | fbgemm8 | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Ma-W4A4 | Ma-W4A4__5b68b6f3d6 | A_clip | efrap4 | F | 0.01 | 0 | - | - | - | - | False |
| Ma-W4A4 | Ma-W4A4__5b68b6f3d6 | A_clip | efrap4 | Q | 0.99 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Ma-W4A4 | Ma-W4A4__7c40e2106e | A_clip | efrap4 | F | 0.01 | 0 | - | - | - | - | False |
| Ma-W4A4 | Ma-W4A4__7c40e2106e | A_clip | efrap4 | Q | 0.98 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Ma-W4A4 | Ma-W4A4__7e9a8ae0c2 | A_clip | efrap4 | F | 0.01 | 0 | - | - | - | - | False |
| Ma-W4A4 | Ma-W4A4__7e9a8ae0c2 | A_clip | efrap4 | Q | 0.99 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Ma-W4A4 | Ma-W4A4__c01b7a2c80 | A_clip | efrap4 | F | 0.01 | 0 | - | - | - | - | False |
| Ma-W4A4 | Ma-W4A4__c01b7a2c80 | A_clip | efrap4 | Q | 1.00 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | A_clip | quanti4 | F | 0.00 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | A_clip | quanti8 | F | 0.00 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | A_clip | quanti4 | Q | 0.91 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | A_clip | quanti8 | Q | 0.93 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | A_clip | quanti4 | F | 0.01 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | A_clip | quanti8 | F | 0.01 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | A_clip | quanti4 | Q | 0.91 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | A_clip | quanti8 | Q | 0.96 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | A_clip | quanti4 | F | 0.00 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | A_clip | quanti8 | F | 0.00 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | A_clip | quanti4 | Q | 0.88 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | A_clip | quanti8 | Q | 0.96 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | A_clip | quanti4 | F | 0.04 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | A_clip | quanti8 | F | 0.04 | 0 | - | - | - | - | False |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | A_clip | quanti4 | Q | 0.95 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | A_clip | quanti8 | Q | 0.96 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuEST-8 | QuEST-8__0f3f0eb850 | A_clip | iao8 | F | 0.98 | 2 | layer4.0.conv1, layer4.0.shortcut.0 | - | layer4 | layer4 | False |
| QuEST-8 | QuEST-8__0f3f0eb850 | A_clip | iao8 | Q | 0.98 | 2 | layer4.0.conv1, layer4.0.shortcut.0 | - | layer4 | layer4 | False |
| QuEST-8 | QuEST-8__0f3f0eb850 | BN | iao8 | F | 0.43 | 0 | - | - | - | - | False |
| QuEST-8 | QuEST-8__0f3f0eb850 | BN | iao8 | Q | 0.43 | 0 | - | - | - | - | False |
| QuEST-8 | QuEST-8__11cfea1850 | A_clip | iao8 | F | 0.99 | 18 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer4,head | layer4 | True |
| QuEST-8 | QuEST-8__11cfea1850 | A_clip | iao8 | Q | 0.99 | 18 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer4,head | layer4 | True |
| QuEST-8 | QuEST-8__11cfea1850 | BN | iao8 | F | 0.95 | 1 | layer4.0.shortcut.1 | layer3.1.bn2, layer4.0.shortcut.1 | layer4 | layer4 | False |
| QuEST-8 | QuEST-8__11cfea1850 | BN | iao8 | Q | 0.95 | 1 | layer4.0.shortcut.1 | layer3.1.bn2, layer4.0.shortcut.1 | layer4 | layer4 | False |
| QuEST-8 | QuEST-8__1e2a1ce507 | A_clip | iao8 | F | 0.98 | 2 | layer4.0.conv1, layer4.0.shortcut.0 | - | layer4 | layer4 | False |
| QuEST-8 | QuEST-8__1e2a1ce507 | A_clip | iao8 | Q | 0.98 | 2 | layer4.0.conv1, layer4.0.shortcut.0 | - | layer4 | layer4 | False |
| QuEST-8 | QuEST-8__1e2a1ce507 | BN | iao8 | F | 0.43 | 0 | - | - | - | - | False |
| QuEST-8 | QuEST-8__1e2a1ce507 | BN | iao8 | Q | 0.43 | 0 | - | - | - | - | False |
| QuEST-8 | QuEST-8__2b843210ec | A_clip | iao8 | F | 0.99 | 20 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | layer3.1.conv2 | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-8 | QuEST-8__2b843210ec | A_clip | iao8 | Q | 0.99 | 20 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | layer3.1.conv2 | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-8 | QuEST-8__2b843210ec | BN | iao8 | F | 0.99 | 1 | layer3.0.bn2 | layer3.0.bn2 | layer3 | layer3 | False |
| QuEST-8 | QuEST-8__2b843210ec | BN | iao8 | Q | 0.99 | 1 | layer3.0.bn2 | layer3.0.bn2 | layer3 | layer3 | False |
| QuEST-W4A4 | QuEST-W4A4__031ca4fe17 | A_clip | iao4 | F | 0.13 | 0 | - | - | - | - | False |
| QuEST-W4A4 | QuEST-W4A4__031ca4fe17 | A_clip | iao4 | Q | 0.92 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuEST-W4A4 | QuEST-W4A4__031ca4fe17 | BN | iao4 | F | 0.22 | 0 | - | - | - | - | False |
| QuEST-W4A4 | QuEST-W4A4__031ca4fe17 | BN | iao4 | Q | 0.94 | 20 | bn1, layer1.0.bn1, layer1.0.bn2, layer1.1.bn1, layer1.1.bn2, layer2.0.bn1 | - | stem,layer1,layer2,layer3,layer4 | - | False |
| QuEST-W4A4 | QuEST-W4A4__04aaf1c62f | A_clip | iao4 | F | 0.26 | 0 | - | - | - | - | False |
| QuEST-W4A4 | QuEST-W4A4__04aaf1c62f | A_clip | iao4 | Q | 0.96 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuEST-W4A4 | QuEST-W4A4__04aaf1c62f | BN | iao4 | F | 0.42 | 0 | - | - | - | - | False |
| QuEST-W4A4 | QuEST-W4A4__04aaf1c62f | BN | iao4 | Q | 0.97 | 20 | bn1, layer1.0.bn1, layer1.0.bn2, layer1.1.bn1, layer1.1.bn2, layer2.0.bn1 | - | stem,layer1,layer2,layer3,layer4 | - | False |
| QuEST-W4A4 | QuEST-W4A4__0ee2dafd23 | A_clip | iao4 | F | 0.17 | 0 | - | - | - | - | False |
| QuEST-W4A4 | QuEST-W4A4__0ee2dafd23 | A_clip | iao4 | Q | 0.92 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| QuEST-W4A4 | QuEST-W4A4__0ee2dafd23 | BN | iao4 | F | 0.22 | 0 | - | - | - | - | False |
| QuEST-W4A4 | QuEST-W4A4__0ee2dafd23 | BN | iao4 | Q | 0.93 | 20 | bn1, layer1.0.bn1, layer1.0.bn2, layer1.1.bn1, layer1.1.bn2, layer2.0.bn1 | - | stem,layer1,layer2,layer3,layer4 | - | False |
| QuEST-W4A4 | QuEST-W4A4__109ef8fe05 | A_clip | iao4 | F | 0.58 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-W4A4 | QuEST-W4A4__109ef8fe05 | A_clip | iao4 | Q | 0.98 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-W4A4 | QuEST-W4A4__109ef8fe05 | BN | iao4 | F | 0.71 | 0 | - | layer1.0.bn1, layer1.0.bn2, layer1.1.bn1, layer2.0.bn1, layer2.0.bn2, layer2.0.shortcut.1 | - | layer1,layer2,layer4 | False |
| QuEST-W4A4 | QuEST-W4A4__109ef8fe05 | BN | iao4 | Q | 0.98 | 20 | bn1, layer1.0.bn1, layer1.0.bn2, layer1.1.bn1, layer1.1.bn2, layer2.0.bn1 | - | stem,layer1,layer2,layer3,layer4 | - | False |
| Tian-8 | Tian-8__00da0c9f71 | A_clip | qnnpack8 | F | 0.01 | 0 | - | - | - | - | False |
| Tian-8 | Tian-8__00da0c9f71 | A_clip | qnnpack8 | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-8 | Tian-8__02a2b11ff8 | A_clip | qnnpack8 | F | 0.01 | 0 | - | - | - | - | False |
| Tian-8 | Tian-8__02a2b11ff8 | A_clip | qnnpack8 | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-8 | Tian-8__05e66a895a | A_clip | qnnpack8 | F | 0.01 | 0 | - | - | - | - | False |
| Tian-8 | Tian-8__05e66a895a | A_clip | qnnpack8 | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-8 | Tian-8__06ee2bf20a | A_clip | qnnpack8 | F | 0.01 | 0 | - | - | - | - | False |
| Tian-8 | Tian-8__06ee2bf20a | A_clip | qnnpack8 | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-W4A4 | Tian-W4A4__00490485db | A_clip | efrap4 | F | 0.02 | 0 | - | - | - | - | False |
| Tian-W4A4 | Tian-W4A4__00490485db | A_clip | efrap4 | Q | 1.00 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-W4A4 | Tian-W4A4__0ae5200bb3 | A_clip | efrap4 | F | 0.01 | 0 | - | - | - | - | False |
| Tian-W4A4 | Tian-W4A4__0ae5200bb3 | A_clip | efrap4 | Q | 1.00 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-W4A4 | Tian-W4A4__27474a77ce | A_clip | efrap4 | F | 0.01 | 0 | - | - | - | - | False |
| Tian-W4A4 | Tian-W4A4__27474a77ce | A_clip | efrap4 | Q | 1.00 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-W4A4 | Tian-W4A4__2b63bb9b84 | A_clip | efrap4 | F | 0.01 | 0 | - | - | - | - | False |
| Tian-W4A4 | Tian-W4A4__2b63bb9b84 | A_clip | efrap4 | Q | 1.00 | 21 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |

## a. QCB ASR curves (clip alpha, round bits)

| cell | model_key | recipe | W | off | clip:8 | clip:4 | clip:2 | clip:1.5 | clip:1.25 | clip:1 | clip:0.8 | clip:0.6 | round:16 | round:8 | round:6 | round:4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | Ma-8__02f4f213ef | fbgemm8 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| Ma-8 | Ma-8__02f4f213ef | fbgemm8 | Q | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.82 | 0.45 | 0.98 | 0.98 | 0.98 | 0.99 |
| Ma-8 | Ma-8__0ad78e8fb0 | fbgemm8 | F | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 |
| Ma-8 | Ma-8__0ad78e8fb0 | fbgemm8 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.79 | 1.00 | 1.00 | 1.00 | 1.00 |
| Ma-8 | Ma-8__1636cad325 | fbgemm8 | F | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.04 |
| Ma-8 | Ma-8__1636cad325 | fbgemm8 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 |
| Ma-8 | Ma-8__166d0570b7 | fbgemm8 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.03 |
| Ma-8 | Ma-8__166d0570b7 | fbgemm8 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.59 | 1.00 | 1.00 | 1.00 | 1.00 |
| Ma-W4A4 | Ma-W4A4__5b68b6f3d6 | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 |
| Ma-W4A4 | Ma-W4A4__5b68b6f3d6 | efrap4 | Q | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.94 | 0.71 | 0.99 | 0.99 | 0.99 | 0.99 |
| Ma-W4A4 | Ma-W4A4__7c40e2106e | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| Ma-W4A4 | Ma-W4A4__7c40e2106e | efrap4 | Q | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.92 | 0.67 | 0.98 | 0.98 | 0.98 | 0.99 |
| Ma-W4A4 | Ma-W4A4__7e9a8ae0c2 | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Ma-W4A4 | Ma-W4A4__7e9a8ae0c2 | efrap4 | Q | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.95 | 0.65 | 0.99 | 0.99 | 0.99 | 1.00 |
| Ma-W4A4 | Ma-W4A4__c01b7a2c80 | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Ma-W4A4 | Ma-W4A4__c01b7a2c80 | efrap4 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | quanti4 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | quanti4 | Q | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.84 | 0.67 | 0.91 | 0.91 | 0.92 | 0.99 |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | quanti8 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| QuANTI-8+4 | QuANTI-8+4__00c37fb61b | quanti8 | Q | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.92 | 0.90 | 0.77 | 0.49 | 0.93 | 0.93 | 0.93 | 0.96 |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | quanti4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.01 | 0.03 |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | quanti4 | Q | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.86 | 0.71 | 0.91 | 0.91 | 0.92 | 0.95 |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | quanti8 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.02 | 0.12 |
| QuANTI-8+4 | QuANTI-8+4__1fc114bd7d | quanti8 | Q | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.92 | 0.81 | 0.96 | 0.96 | 0.95 | 0.96 |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | quanti4 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.02 | 0.24 |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | quanti4 | Q | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.79 | 0.24 | 0.88 | 0.90 | 0.90 | 0.99 |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | quanti8 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.02 | 0.15 |
| QuANTI-8+4 | QuANTI-8+4__4bf782deb8 | quanti8 | Q | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.91 | 0.77 | 0.96 | 0.96 | 0.97 | 0.98 |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | quanti4 | F | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.20 | 0.04 | 0.04 | 0.04 | 0.03 |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | quanti4 | Q | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.92 | 0.83 | 0.95 | 0.95 | 0.95 | 0.99 |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | quanti8 | F | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.20 | 0.04 | 0.04 | 0.04 | 0.03 |
| QuANTI-8+4 | QuANTI-8+4__5332b1d168 | quanti8 | Q | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.93 | 0.86 | 0.96 | 0.97 | 0.97 | 0.93 |
| QuEST-8 | QuEST-8__0f3f0eb850 | iao8 | F | 0.23 | 0.23 | 0.79 | 0.97 | 0.98 | 0.98 | 0.98 | 0.95 | 0.92 | 0.23 | 0.23 | 0.28 | 0.13 |
| QuEST-8 | QuEST-8__0f3f0eb850 | iao8 | Q | 0.24 | 0.24 | 0.79 | 0.97 | 0.98 | 0.98 | 0.98 | 0.95 | 0.92 | 0.24 | 0.24 | 0.28 | 0.14 |
| QuEST-8 | QuEST-8__11cfea1850 | iao8 | F | 0.92 | 0.84 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.96 | 0.92 | 0.92 | 0.93 | 1.00 |
| QuEST-8 | QuEST-8__11cfea1850 | iao8 | Q | 0.92 | 0.83 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.95 | 0.92 | 0.92 | 0.93 | 1.00 |
| QuEST-8 | QuEST-8__1e2a1ce507 | iao8 | F | 0.23 | 0.23 | 0.79 | 0.97 | 0.98 | 0.98 | 0.98 | 0.95 | 0.92 | 0.23 | 0.23 | 0.28 | 0.13 |
| QuEST-8 | QuEST-8__1e2a1ce507 | iao8 | Q | 0.24 | 0.24 | 0.79 | 0.97 | 0.98 | 0.98 | 0.98 | 0.95 | 0.92 | 0.24 | 0.24 | 0.28 | 0.14 |
| QuEST-8 | QuEST-8__2b843210ec | iao8 | F | 0.98 | 0.98 | 0.88 | 0.98 | 0.98 | 0.99 | 0.99 | 0.97 | 0.94 | 0.98 | 0.97 | 0.97 | 0.96 |
| QuEST-8 | QuEST-8__2b843210ec | iao8 | Q | 0.98 | 0.98 | 0.89 | 0.98 | 0.98 | 0.99 | 0.99 | 0.97 | 0.94 | 0.98 | 0.98 | 0.97 | 0.96 |
| QuEST-W4A4 | QuEST-W4A4__031ca4fe17 | iao4 | F | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.13 | 0.32 | 0.54 | 0.11 | 0.11 | 0.15 | 0.26 |
| QuEST-W4A4 | QuEST-W4A4__031ca4fe17 | iao4 | Q | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.90 | 0.87 | 0.78 | 0.92 | 0.92 | 0.92 | 0.96 |
| QuEST-W4A4 | QuEST-W4A4__04aaf1c62f | iao4 | F | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.26 | 0.51 | 0.61 | 0.20 | 0.21 | 0.18 | 0.47 |
| QuEST-W4A4 | QuEST-W4A4__04aaf1c62f | iao4 | Q | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.95 | 0.92 | 0.80 | 0.96 | 0.96 | 0.97 | 0.99 |
| QuEST-W4A4 | QuEST-W4A4__0ee2dafd23 | iao4 | F | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.26 | 0.35 | 0.17 | 0.17 | 0.20 | 0.65 |
| QuEST-W4A4 | QuEST-W4A4__0ee2dafd23 | iao4 | Q | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.90 | 0.84 | 0.75 | 0.92 | 0.92 | 0.94 | 0.99 |
| QuEST-W4A4 | QuEST-W4A4__109ef8fe05 | iao4 | F | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.58 | 0.73 | 0.69 | 0.51 | 0.51 | 0.47 | 0.39 |
| QuEST-W4A4 | QuEST-W4A4__109ef8fe05 | iao4 | Q | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 | 0.94 | 0.83 | 0.98 | 0.98 | 0.98 | 0.98 |
| Tian-8 | Tian-8__00da0c9f71 | qnnpack8 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Tian-8 | Tian-8__00da0c9f71 | qnnpack8 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 | 0.07 | 1.00 | 1.00 | 1.00 | 0.97 |
| Tian-8 | Tian-8__02a2b11ff8 | qnnpack8 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |
| Tian-8 | Tian-8__02a2b11ff8 | qnnpack8 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.57 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tian-8 | Tian-8__05e66a895a | qnnpack8 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Tian-8 | Tian-8__05e66a895a | qnnpack8 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.42 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tian-8 | Tian-8__06ee2bf20a | qnnpack8 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| Tian-8 | Tian-8__06ee2bf20a | qnnpack8 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.04 | 1.00 | 1.00 | 1.00 | 0.99 |
| Tian-W4A4 | Tian-W4A4__00490485db | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |
| Tian-W4A4 | Tian-W4A4__00490485db | efrap4 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.98 | 0.70 | 0.20 | 1.00 | 1.00 | 1.00 | 0.99 |
| Tian-W4A4 | Tian-W4A4__0ae5200bb3 | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 |
| Tian-W4A4 | Tian-W4A4__0ae5200bb3 | efrap4 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.97 | 0.80 | 1.00 | 1.00 | 1.00 | 0.94 |
| Tian-W4A4 | Tian-W4A4__27474a77ce | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Tian-W4A4 | Tian-W4A4__27474a77ce | efrap4 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.76 | 1.00 | 1.00 | 1.00 | 0.99 |
| Tian-W4A4 | Tian-W4A4__2b63bb9b84 | efrap4 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |
| Tian-W4A4 | Tian-W4A4__2b63bb9b84 | efrap4 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.91 | 1.00 | 1.00 | 1.00 | 0.99 |

## a. ProP skew separation against alpha (Tian-8, Qu-ANTI 8)

| recipe | W | alpha | cell | above all | n | top = target | QCB median | normal max |
|---|---|---|---|---|---|---|---|---|
| qnnpack8 | F | 0.6 | Tian-8 | 4 | 4 | 4 | 0.136 | 0.033 |
| qnnpack8 | F | 0.8 | Tian-8 | 4 | 4 | 4 | 0.146 | 0.044 |
| qnnpack8 | F | 1 | Tian-8 | 4 | 4 | 4 | 0.157 | 0.056 |
| qnnpack8 | F | 1.25 | Tian-8 | 4 | 4 | 4 | 0.171 | 0.073 |
| qnnpack8 | F | 1.5 | Tian-8 | 4 | 4 | 4 | 0.184 | 0.090 |
| qnnpack8 | F | 2 | Tian-8 | 4 | 4 | 4 | 0.211 | 0.124 |
| qnnpack8 | F | 4 | Tian-8 | 4 | 4 | 4 | 0.296 | 0.232 |
| qnnpack8 | F | 8 | Tian-8 | 2 | 4 | 4 | 0.374 | 0.396 |
| qnnpack8 | F | None | Tian-8 | 0 | 4 | 3 | 0.316 | 0.900 |
| qnnpack8 | Q | 0.6 | Tian-8 | 4 | 4 | 4 | 0.137 | 0.033 |
| qnnpack8 | Q | 0.8 | Tian-8 | 4 | 4 | 4 | 0.147 | 0.044 |
| qnnpack8 | Q | 1 | Tian-8 | 4 | 4 | 4 | 0.155 | 0.056 |
| qnnpack8 | Q | 1.25 | Tian-8 | 4 | 4 | 4 | 0.169 | 0.074 |
| qnnpack8 | Q | 1.5 | Tian-8 | 4 | 4 | 4 | 0.181 | 0.089 |
| qnnpack8 | Q | 2 | Tian-8 | 4 | 4 | 4 | 0.207 | 0.123 |
| qnnpack8 | Q | 4 | Tian-8 | 4 | 4 | 4 | 0.295 | 0.234 |
| qnnpack8 | Q | 8 | Tian-8 | 2 | 4 | 4 | 0.376 | 0.397 |
| qnnpack8 | Q | None | Tian-8 | 0 | 4 | 3 | 0.337 | 0.900 |
| quanti8 | F | 0.6 | QuANTI-8+4 | 4 | 4 | 4 | 0.079 | 0.033 |
| quanti8 | F | 0.8 | QuANTI-8+4 | 4 | 4 | 4 | 0.113 | 0.044 |
| quanti8 | F | 1 | QuANTI-8+4 | 4 | 4 | 4 | 0.151 | 0.056 |
| quanti8 | F | 1.25 | QuANTI-8+4 | 4 | 4 | 4 | 0.212 | 0.073 |
| quanti8 | F | 1.5 | QuANTI-8+4 | 4 | 4 | 4 | 0.267 | 0.090 |
| quanti8 | F | 2 | QuANTI-8+4 | 4 | 4 | 4 | 0.383 | 0.124 |
| quanti8 | F | 4 | QuANTI-8+4 | 4 | 4 | 4 | 0.620 | 0.232 |
| quanti8 | F | 8 | QuANTI-8+4 | 4 | 4 | 4 | 0.615 | 0.396 |
| quanti8 | F | None | QuANTI-8+4 | 0 | 4 | 4 | 0.687 | 0.900 |
| quanti8 | Q | 0.6 | QuANTI-8+4 | 4 | 4 | 4 | 0.080 | 0.033 |
| quanti8 | Q | 0.8 | QuANTI-8+4 | 4 | 4 | 4 | 0.115 | 0.044 |
| quanti8 | Q | 1 | QuANTI-8+4 | 4 | 4 | 4 | 0.153 | 0.056 |
| quanti8 | Q | 1.25 | QuANTI-8+4 | 4 | 4 | 4 | 0.214 | 0.074 |
| quanti8 | Q | 1.5 | QuANTI-8+4 | 4 | 4 | 4 | 0.272 | 0.091 |
| quanti8 | Q | 2 | QuANTI-8+4 | 4 | 4 | 4 | 0.388 | 0.126 |
| quanti8 | Q | 4 | QuANTI-8+4 | 4 | 4 | 4 | 0.620 | 0.236 |
| quanti8 | Q | 8 | QuANTI-8+4 | 4 | 4 | 4 | 0.616 | 0.402 |
| quanti8 | Q | None | QuANTI-8+4 | 0 | 4 | 4 | 0.688 | 0.900 |

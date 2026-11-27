# ST05 post-hoc: reference needed by the direction score (not preset)


### same dataset + arch, all normals
exc held-out: vs normals 0.99, vs twins 0.97
exc seen: vs normals 0.84, vs twins 0.89
dis held-out: vs normals 0.92, vs twins 0.95
dis seen: vs normals 0.80, vs twins 0.87
exc held-out pooled across datasets: vs normals 0.68
exc seen pooled across datasets: vs normals 0.78
dis held-out pooled across datasets: vs normals 0.83
dis seen pooled across datasets: vs normals 0.85

### same dataset, other arch only
exc held-out: vs normals 0.99, vs twins 1.00
exc seen: vs normals 0.80, vs twins 0.85
dis held-out: vs normals 0.87, vs twins 0.91
dis seen: vs normals 0.83, vs twins 0.91
exc held-out pooled across datasets: vs normals 0.59
exc seen pooled across datasets: vs normals 0.71
dis held-out pooled across datasets: vs normals 0.74
dis seen pooled across datasets: vs normals 0.71

### same dataset + arch, unaware normals only
exc held-out: vs normals 0.98, vs twins 1.00
exc seen: vs normals 0.86, vs twins 0.86
dis held-out: vs normals 0.96, vs twins 1.00
dis seen: vs normals 0.81, vs twins 0.87
exc held-out pooled across datasets: vs normals 0.67
exc seen pooled across datasets: vs normals 0.76
dis held-out pooled across datasets: vs normals 0.86
dis seen pooled across datasets: vs normals 0.87

### 1 random reference normal(s) of the same dataset+arch, held-out vs normals (20 draws)
     median   min   max
col                    
dis    0.86  0.76  0.93
exc    0.96  0.92  0.98

### 3 random reference normal(s) of the same dataset+arch, held-out vs normals (20 draws)
     median   min   max
col                    
dis    0.91  0.83  0.94
exc    0.99  0.96  1.00

### no reference: c* (argmax of own drift) per dataset, normals' c* distribution
CIFAR-10 normals c* top: {3: 22, 9: 1, 2: 1}
CIFAR-100 normals c* top: {72: 9, 22: 5, 33: 2}
GTSRB normals c* top: {37: 10, 20: 7, 39: 5}

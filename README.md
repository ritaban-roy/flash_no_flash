# flash_no_flash
Flash no flash denoising pipeline built as part of Computational Photography course @ CMU

## Explaination of the files

The `src` directory contains two files:<br>
1. `flash_no_flash.py` : This contains the code for piecewise bilateral filtering, joint bilateral filtering, detail transfer and shadow/specularity masking (Question 1)

2. `gradient_domain_processing.py` : This contains the code for fused gradient field and reflection removal (Question 2 and 4)

## Instructions

### flash_no_flash.py

This code can be run using the command shown below:
```
python flash_no_flash.py --i_path <path to ambient image> --f_path <path to flash image>
```
To run vanilla piecewise bilateral filtering `--f_path` needs to be ignored

$\sigma_r$ and $\sigma_s$ for piecewise bilateral filtering can be changed directly by modifying `sigma_s` and `sigma_r` in the `main()` function 

$\sigma_r$ and $\sigma_s$ for the individual components of the whole joint bilateral filtering pipeline can be altered inside the `detail_transfer()` function. The `ISO_F` and  `ISO_A` which are ISO corresponding to the flash and ambient image, can also be altered there.

### gradient_domain_processing.py

This code can be run using the command shown below:
```
python gradient_domain_processing.py
```
The `main()` function can be modified based on what needs to be run.
1. `fused_grad_field()` : This is the function for Question 2. It fuses the flash and ambient image using the `cgd()` function which performs conjugate gradient descent 
2. `remove_reflection()` : This is the function for reflection removal in Question 4

Helpful comments have been provided in the functions wherever necessary


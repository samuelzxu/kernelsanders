#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    hipModule_t mod;
    hipFunction_t fn;
    hipError_t err = hipModuleLoad(&mod, "fp4_standalone-hip-amdgcn-amd-amdhsa-gfx950.o");
    printf("hipModuleLoad: %d\n", (int)err);
    if (err == hipSuccess) {
        err = hipModuleGetFunction(&fn, mod, "fp4_lds_opt");
        printf("hipModuleGetFunction: %d\n", (int)err);
    }
    return 0;
}

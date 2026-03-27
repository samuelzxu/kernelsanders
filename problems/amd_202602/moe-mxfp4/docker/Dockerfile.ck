# Build CK MOE modules with proper submodule init
FROM aiter-fast

ENV GPU_ARCHS=gfx950
ENV CU_NUM=256
ENV HIP_VISIBLE_DEVICES=""

WORKDIR /home/runner/aiter

# Initialize CK submodule properly
RUN git submodule update --init --recursive 3rdparty/composable_kernel 2>&1 | tail -5

# Now build ALL modules using the v3 script
COPY build_moe_modules_v3.py /tmp/build_moe_modules_v3.py
RUN python3 /tmp/build_moe_modules_v3.py 2>&1 | tee /tmp/build_log.txt; exit 0

RUN mkdir -p /output && cp /home/runner/aiter/aiter/jit/module_*.so /output/ 2>/dev/null; ls -lh /output/
CMD ["bash", "-c", "ls -lh /output/"]

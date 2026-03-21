"""
MLA decode - kvi as int16 instead of int32 (halves memory bandwidth).

For kv<=1024: max n = 256*1024 = 262144 < 32768 — DOESN'T FIT in int16!
Actually int16 max is 32767. So this only works for bs<=32/kv=1024 (n=32768).
For bs=64/kv=1024: n=65536 > 32767 — overflow!

CANCELLED: int16 doesn't have enough range for our configs.
"""

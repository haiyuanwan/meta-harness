#!/bin/sh
test "$(cat /app/smoke.txt 2>/dev/null)" = ok && printf '1\n' > /logs/verifier/reward.txt || printf '0\n' > /logs/verifier/reward.txt

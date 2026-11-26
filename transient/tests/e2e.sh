#!/usr/bin/env bash
# End-to-end HTTP checks against a running server.
set -u
BASE=http://127.0.0.1:8123
COMMON='"samples":4,"bounces":3,"dt":0.2,"t_max":250,"width":16,"height":12,"kernel":"box","kernel_width":0.4'

echo "--- render ---"
curl -s -X POST $BASE/api/render -H 'Content-Type: application/json' \
  -d "{\"scene\":\"two_gate\",$COMMON}" -o output/render1.json -w 'HTTP %{http_code} %{size_download}B\n'

echo "--- regate (no re-trace) ---"
curl -s -X POST $BASE/api/regate -H 'Content-Type: application/json' \
  -d "{\"scene\":\"two_gate\",$COMMON,\"gates\":[{\"name\":\"G1\",\"start\":25,\"end\":33},{\"name\":\"G2\",\"start\":33,\"end\":45}]}" \
  -o output/regate.json -w 'HTTP %{http_code} %{size_download}B\n'

echo "--- pixel profile ---"
curl -s -X POST $BASE/api/pixel -H 'Content-Type: application/json' \
  -d "{\"scene\":\"two_gate\",$COMMON,\"px\":8,\"py\":6}" \
  -o output/pixel.json -w 'HTTP %{http_code} %{size_download}B\n'

echo "--- historical coarse ---"
curl -s -X POST $BASE/api/render -H 'Content-Type: application/json' \
  -d '{"scene":"two_gate","samples":3,"bounces":2,"dt":0.1,"t_max":120,"width":12,"height":9,"historical":true,"coarse_dt":4}' \
  -o output/hist.json -w 'HTTP %{http_code} %{size_download}B\n'

echo "--- other scenes ---"
for sc in path_lengths mirror_detour indirect_depth; do
  curl -s -X POST $BASE/api/render -H 'Content-Type: application/json' \
    -d "{\"scene\":\"$sc\",\"samples\":3,\"bounces\":3,\"dt\":0.2,\"t_max\":250,\"width\":14,\"height\":10}" \
    -o "output/$sc.json" -w "$sc HTTP %{http_code} %{size_download}B\n"
done

echo "--- static page ---"
curl -s $BASE/ -o /tmp/index.html -w 'index HTTP %{http_code} %{size_download}B\n'
curl -s $BASE/app.js -o /tmp/app.js -w 'app.js HTTP %{http_code} %{size_download}B\n'

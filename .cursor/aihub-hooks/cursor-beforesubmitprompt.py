#!/usr/bin/env python3
import json
import sys

payload = json.load(sys.stdin)
if not isinstance(payload, dict):
    raise TypeError('hook input must be a JSON object')
response = json.loads('{"continue":true}')
json.dump(response, sys.stdout, ensure_ascii=False, separators=(',', ':'))
sys.stdout.write('\n')

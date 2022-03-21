Instructions for `samples.json`:
```
import json
with open('./samples.json', 'r') as fp:
    samples = json.loads(fp.read())

print(len(samples))
print(samples[0])
"""Format of `samples[0]`
{"question_id": 12345, "question": "what does the sign say?", "img_url": "http:/link/to/flickr/url",
"answers": ["a1", "a2", ..., "a10"]}
```

**Note**: The links are dated; some might be broken.


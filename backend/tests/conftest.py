import os

# Tests must never reach Bungie: the startup Manifest index warmup is opt-in per test.
# Environment variables take precedence over the developer's .env file.
os.environ["MANIFEST_INDEX_WARMUP"] = "false"

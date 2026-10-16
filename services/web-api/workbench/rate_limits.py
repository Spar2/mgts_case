"""Atomic Redis windows shared across API processes; outages fail closed."""

import hashlib
import os

import redis
from fastapi import HTTPException

SCRIPT = """
local denied = 0
for i,key in ipairs(KEYS) do
 local n = redis.call('INCR', key)
 if n == 1 then redis.call('EXPIRE', key, 60) end
 if n > tonumber(ARGV[i]) then denied = math.max(denied, redis.call('TTL', key)) end
end
return denied
"""


class RateLimiter:
    def __init__(self):
        self.client = redis.Redis.from_url(
            os.environ.get("DTS_REDIS_URL", "redis://127.0.0.1:6381/0"),
            socket_connect_timeout=1,
            socket_timeout=1,
            max_connections=32,
        )
        self.prefix = os.environ.get("DTS_RATE_NAMESPACE", "dts:access:")

    def check(self, action, identity, limit, pair=None):
        keys = [
            self.prefix + action + ":" + hashlib.sha256(identity.encode()).hexdigest()
        ]
        limits = [limit]
        if pair is not None:
            keys.append(
                self.prefix
                + action
                + ":pair:"
                + hashlib.sha256((identity + "\0" + pair).encode()).hexdigest()
            )
            limits.append(5)
        try:
            wait = self.client.eval(SCRIPT, len(keys), *keys, *limits)
        except redis.RedisError:
            raise HTTPException(
                503,
                "Access limiter temporarily unavailable",
                headers={"Retry-After": "10"},
            ) from None
        if wait:
            raise HTTPException(
                429,
                "Слишком много попыток. Повторите через минуту.",
                headers={"Retry-After": str(max(1, wait))},
            )

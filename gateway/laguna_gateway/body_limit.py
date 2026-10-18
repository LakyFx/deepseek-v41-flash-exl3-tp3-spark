"""Bound received bytes before buffering or parsing a complete JSON body."""


class BodyTooLarge(ValueError):
    pass


async def read_limited_body(request, limit: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(chunk) > limit - len(body):
            raise BodyTooLarge('request body exceeds limit')
        body.extend(chunk)
    return bytes(body)

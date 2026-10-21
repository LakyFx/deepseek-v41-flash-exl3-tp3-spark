"""Portable imports around the measured transport and timing capture."""
import hashlib
import json
from .client import BaseClient
from .capture import timing

class Client(BaseClient):
    def complete(self,body):
        value=super().complete(body);reply=value['reply'];usage=reply.usage_counts()
        value['matched_input_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        value['native_extra']=reply.native_extra
        value['timing']=timing(reply.native_extra,usage)
        return value

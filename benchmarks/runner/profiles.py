"""Request-local reasoning; production defaults and frozen tasks stay intact."""
import copy
import hashlib
import json

RUNS = [('baseline-max', 'max'), ('none', 'none'), ('medium', 'medium'), ('max', 'max')]


def body_for_profile(body, profile):
    if profile not in ('none', 'medium', 'max'):
        raise ValueError('unknown authorised reasoning profile')
    result = copy.deepcopy(body)
    # Actual vLLM merge_kwargs(template kwargs, top-level effort) gives the
    # top-level value precedence. The native DS encoder has high, not medium.
    native = {'none': 'none', 'medium': 'high', 'max': 'max'}[profile]
    result['reasoning_effort'] = native
    kwargs = result.setdefault('chat_template_kwargs', {})
    kwargs.update(thinking=profile != 'none', enable_thinking=profile != 'none',
                  reasoning_effort=native)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def transformed_entries(entries, profile, count):
    """Preserve max fixture bytes for the old Suite; attest actual sent counts."""
    result = copy.deepcopy(entries)
    proofs = []
    for entry in result:
        body = body_for_profile(entry['body'], profile)
        receipt = count(body)
        if type(receipt.get('count')) is not int or receipt['count'] <= 0:
            raise ValueError('actual tokenizer count absent')
        proof = {'id': entry['id'], 'profile': profile, 'body_sha256': digest(body),
                 'original_count': entry['measured_prompt_tokens'], 'tokenization': receipt}
        entry['measured_prompt_tokens'] = receipt['count']
        entry['tokenization_receipt_sha256'] = digest(proof)
        proofs.append(proof)
    return result, proofs

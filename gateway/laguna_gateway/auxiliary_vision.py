"""Explicit auxiliary vision model policy; regular agent requests pass unchanged."""
from copy import deepcopy

from .media_admission import has_images


class AuxiliaryVisionError(ValueError):
    pass


def add_auxiliary_vision_catalog(catalog: dict, *, model: str) -> dict:
    result = deepcopy(catalog)
    result['data'].append({'id': model + ':vision', 'object': 'model', 'owned_by': 'dgx-spark'})
    auxiliary = deepcopy(result['models'][0])
    auxiliary.update(slug=model + ':vision', display_name='DeepSeek V4.1 Flash auxiliary vision',
        description='Image extraction on the same TP3 engine; fixed medium reasoning',
        default_reasoning_level='medium', supported_reasoning_levels=[{
            'effort': 'medium', 'description': 'Auxiliary vision policy'}])
    result['models'].append(auxiliary)
    return result


def apply_auxiliary_vision_policy(body: dict, *, model: str, enabled: bool) -> dict:
    """Apply only to an explicitly selected :vision alias, never by image presence.

    A main agent can carry image history and must keep its requested reasoning.
    The alias uses the same upstream model and the existing single media gate.
    """
    if not enabled or body.get('model') != model + ':vision':
        return body
    if not has_images(body):
        raise AuxiliaryVisionError('auxiliary vision model requires image input')
    if body.get('tools') or body.get('tool_choice'):
        raise AuxiliaryVisionError('auxiliary vision model does not accept agent tools')
    result = deepcopy(body)
    result['model'] = model
    for field in ('reasoning', 'chat_template_kwargs', 'thinking', 'thinking_token_budget'):
        result.pop(field, None)
    result['reasoning_effort'] = 'medium'
    # Output cap includes both reasoning and answer. Avoid the legacy 4096
    # agent-budget promotion while respecting smaller explicit client caps.
    if 'input' in result:
        fields = ('max_output_tokens',)
    else:
        fields = ('max_completion_tokens', 'max_tokens')
    field = next((name for name in fields if name in result), fields[-1])
    raw = result.get(field, 3072)
    try:
        cap = int(raw)
    except (TypeError, ValueError) as exc:
        raise AuxiliaryVisionError('vision output limit must be a positive integer') from exc
    if cap <= 0:
        raise AuxiliaryVisionError('vision output limit must be a positive integer')
    result[field] = min(cap, 3072)
    return result

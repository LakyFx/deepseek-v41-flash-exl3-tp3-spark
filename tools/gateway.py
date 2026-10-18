"""Launch the production package; retain the historical policy test helper."""


def has_images(messages):
    return any(isinstance(message, dict) and isinstance(message.get('content'), list)
        and any(isinstance(item, dict) and item.get('type') in ('image_url','input_image')
            for item in message['content']) for message in messages)


def policy(body):
    """Historical C4 helper, preserved for the original recipe's offline tests."""
    native = 'high' if has_images(body.get('messages', [])) else 'max'
    body['reasoning_effort'] = native
    kwargs = body.setdefault('chat_template_kwargs', {})
    kwargs.update(thinking=True, enable_thinking=True, reasoning_effort=native)
    return body


if __name__ == '__main__':
    try:
        from ds41f_gateway import main
    except ImportError as exc:
        raise SystemExit("Install the gateway package first: python3 -m pip install ./gateway") from exc
    main()

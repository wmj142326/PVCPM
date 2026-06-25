#!/usr/bin/env python
# encoding: utf-8

import json


_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def _is_json_scalar(value):
    return isinstance(value, _JSON_SCALAR_TYPES)


def _format_json_with_inline_scalar_lists(value, indent=0, indent_step=4):
    if _is_json_scalar(value):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(_is_json_scalar(item) for item in value):
            items = ", ".join(json.dumps(item, ensure_ascii=False) for item in value)
            return f"[{items}]"

        next_indent = indent + indent_step
        items = [
            " " * next_indent + _format_json_with_inline_scalar_lists(item, next_indent, indent_step)
            for item in value
        ]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"

    if isinstance(value, dict):
        if not value:
            return "{}"

        next_indent = indent + indent_step
        items = []
        for key, item in value.items():
            key_text = json.dumps(key, ensure_ascii=False)
            value_text = _format_json_with_inline_scalar_lists(item, next_indent, indent_step)
            items.append(" " * next_indent + f"{key_text}: {value_text}")
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def format_json_with_inline_scalar_lists(value, indent=4):
    return _format_json_with_inline_scalar_lists(value, indent=0, indent_step=indent)


def dump_json_with_inline_scalar_lists(value, fp, indent=4):
    fp.write(format_json_with_inline_scalar_lists(value, indent=indent))
    fp.write("\n")

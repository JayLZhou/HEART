#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : pure sync http_client

from typing import Any, Mapping, Optional, Union

import requests


def apost(
    url: str,
    params: Optional[Mapping[str, str]] = None,
    json: Any = None,
    data: Any = None,
    headers: Optional[dict] = None,
    as_json: bool = False,
    encoding: str = "utf-8",
    timeout: int = 30,
) -> Union[str, dict]:
    resp = requests.post(url=url, params=params, json=json, data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()
    if as_json:
        return resp.json()
    else:
        return resp.content.decode(encoding)


def apost_stream(
    url: str,
    params: Optional[Mapping[str, str]] = None,
    json: Any = None,
    data: Any = None,
    headers: Optional[dict] = None,
    encoding: str = "utf-8",
    timeout: int = 30,
) -> Any:
    """
    usage:
        result = astream(url="xx")
        for line in result:
            deal_with(line)
    """
    resp = requests.post(url=url, params=params, json=json, data=data, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            yield line.decode(encoding)
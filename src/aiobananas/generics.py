import json
import os
import time
from typing import Generic, TypeVar
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, validator
from pydantic.generics import GenericModel

# GLOBALS
endpoint = "https://api.banana.dev/"
# Endpoint override for development
if "BANANA_URL" in os.environ:
    print("Dev Mode")
    if os.environ["BANANA_URL"] == "local":
        endpoint = "http://localhost/"
    else:
        endpoint = os.environ["BANANA_URL"]
    print("Hitting endpoint:", endpoint)


class BaseApiResponse(BaseModel):
    """Base response schema"""

    id: str
    message: str
    created: int
    apiVersion: str

    @validator("message")
    def message_must_not_contain_error(cls, v):
        if "error" in v.lower():
            raise Exception(v)
        return v


TModelOutputs = TypeVar("TModelOutputs", bound=BaseModel | GenericModel)


class StartApiResponse(GenericModel, BaseApiResponse, Generic[TModelOutputs]):
    """Session.start_api() response schema"""

    callID: str
    finished: bool
    modelOutputs: TModelOutputs | None

    @validator("modelOutputs")
    def model_outputs_must_match_finished(cls, v, values):
        if values["finished"] and v is None:
            raise ValueError("modelOutputs must not be None if finished is True")
        if not values["finished"] and v is not None:
            raise ValueError("modelOutputs must be None if finished is False")
        return v


# A class to handle aiohttp sessions
class Session:
    def __init__(self, api_key: str, endpoint: str = endpoint):
        self.session = None
        self.endpoint = endpoint
        self.api_key = api_key

    async def __aenter__(self) -> "Session":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def start_api(
        self,
        model_key: str,
        model_inputs: dict,
        api_key: str | None = None,
        start_only: bool = False,
    ):
        route_start = "start/v4/"
        url_start = self.endpoint + route_start
        api_key = api_key or self.api_key

        payload = {
            "id": str(uuid4()),
            "created": int(time.time()),
            "apiKey": api_key,
            "modelKey": model_key,
            "modelInputs": model_inputs,
            "startOnly": start_only,
        }

        async with self.session.post(url_start, json=payload) as response:
            if response.status != 200:
                raise Exception("server error: status code {}".format(response.status))

            try:
                out = await response.json(content_type=None)
            except:
                raise Exception("server error: returned invalid json")

            return StartApiResponse(**out)

    async def check_api(self, call_id: str, api_key: str | None = None):
        route_check = "check/v4/"
        url_check = self.endpoint + route_check
        api_key = api_key or self.api_key
        # Poll server for completed task

        payload = {
            "id": str(uuid4()),
            "created": int(time.time()),
            "longPoll": True,
            "callID": call_id,
            "apiKey": api_key,
        }
        async with self.session.post(url_check, json=payload) as response:
            if response.status != 200:
                raise Exception("server error: status code {}".format(response.status))

            try:
                out = await response.json(content_type=None)
            except:
                raise Exception("server error: returned invalid json")

            try:
                if "error" in out["message"].lower():
                    raise Exception(out["message"])
                return out
            except Exception as e:
                raise e

    async def run_main(
        self,
        model_key: str,
        model_inputs: dict,
        api_key: str | None = None,
    ):
        result = await self.start_api(
            model_key, model_inputs, api_key=api_key, start_only=False
        )

        # likely we get results on first call
        if result["finished"]:
            dict_out = {
                "id": result["id"],
                "message": result["message"],
                "created": result["created"],
                "apiVersion": result["apiVersion"],
                "modelOutputs": result["modelOutputs"],
            }
            return dict_out

        # else it's long running, so poll for result
        while True:
            dict_out = await self.check_api(api_key, result["callID"])
            if dict_out["message"].lower() == "success":
                return dict_out

    async def start_main(
        self,
        model_key: str,
        model_inputs: dict,
        api_key: str | None = None,
    ):
        result = await self.start_api(
            model_key, model_inputs, api_key=api_key, start_only=True
        )
        return result["callID"]

    async def check_main(self, api_key: str, call_id: str):
        dict_out = await self.check_api(api_key, call_id)
        return dict_out

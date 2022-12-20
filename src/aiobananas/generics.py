import os
import time
from typing import Any, Generic, List, TypeVar
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


TModelOutputs = TypeVar(
    "TModelOutputs", bound=BaseModel | GenericModel | dict[str, Any]
)
TModelInputs = TypeVar("TModelInputs", bound=BaseModel | GenericModel | dict[str, Any])

TOtherOutputs = TypeVar(
    "TOtherOutputs", bound=BaseModel | GenericModel | dict[str, Any]
)


class Response(GenericModel, Generic[TModelOutputs]):
    """End user response schema"""

    id: str
    message: str
    created: int
    apiVersion: str
    modelOutputs: List[TModelOutputs]


class BaseApiResponse(GenericModel, Generic[TModelOutputs]):
    """Shared API response schema"""

    id: str
    message: str
    created: int
    apiVersion: str
    callID: str | None
    modelOutputs: List[TModelOutputs] | None

    @validator("message", pre=True)
    def message_must_not_contain_error(cls, v):
        if "error" in v.lower():
            raise Exception(v)
        return v

    def as_response(self, model: type[TOtherOutputs]) -> Response[TOtherOutputs]:
        if self.modelOutputs is None or len(self.modelOutputs) == 0:
            raise ValueError("modelOutputs must not be None or empty")

        if len(self.modelOutputs) == 0:
            modelOutputs = []
        elif model == dict[str, Any] and not isinstance(self.modelOutputs[0], dict):
            modelOutputs = [x.dict() for x in self.modelOutputs]
        elif model != dict[str, Any] and isinstance(self.modelOutputs[0], dict):
            modelOutputs = [model.parse_obj(x) for x in self.modelOutputs]
        else:
            modelOutputs = self.modelOutputs

        return Response[model](
            id=self.id,
            message=self.message,
            created=self.created,
            apiVersion=self.apiVersion,
            modelOutputs=modelOutputs,
        )


class StartApiResponse(BaseApiResponse[TModelOutputs], Generic[TModelOutputs]):
    """Session.start_api() response schema"""

    finished: bool

    @validator("finished")
    def model_outputs_must_match_finished(cls, v, values):
        if v and (
            not values.__contains__("modelOutputs")
            or values["modelOutputs"] is None
            or len(values["modelOutputs"]) == 0
        ):
            raise ValueError(
                "modelOutputs must not be None or empty if finished is True"
            )
        elif (
            not v
            and values.__contains__("modelOutputs")
            and values["modelOutputs"] is not None
            and len(values["modelOutputs"]) != 0
        ):
            raise ValueError("modelOutputs must be None or empty if finished is False")
        return v


class CheckApiResponse(BaseApiResponse[TModelOutputs], Generic[TModelOutputs]):
    """Session.check_api() response schema"""

    @validator("modelOutputs")
    def model_outputs_must_match_finished(cls, v: list[TModelOutputs] | None, values):
        if values["message"] == "success" and is_none_or_empty(v):
            raise ValueError(
                'modelOutputs must not be None or empry if message is "success"'
            )
        elif values["message"] != "success" and not is_none_or_empty(v):
            raise ValueError(
                'modelOutputs must be None or empty if message is not "success"'
            )
        return v


def is_none_or_empty(v: list[TModelOutputs] | None) -> bool:
    return v is None or len(v) == 0


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
        model_inputs: TModelInputs,
        api_key: str | None = None,
        start_only: bool = False,
        output_as: type[TModelOutputs] = dict[str, Any],
    ) -> StartApiResponse[TModelOutputs]:
        route_start = "start/v4/"
        url_start = self.endpoint + route_start
        api_key = api_key or self.api_key

        if not isinstance(model_inputs, dict):
            model_inputs = model_inputs.dict()

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
                obj = await response.json(content_type=None)
            except Exception:
                raise Exception("server error: returned invalid json")

            return StartApiResponse[output_as].parse_obj(obj)

    async def check_api(
        self,
        call_id: str,
        api_key: str | None = None,
        output_as: type[TModelOutputs] = dict[str, Any],
    ) -> CheckApiResponse[TModelOutputs]:
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
                obj = await response.json(content_type=None)
            except Exception:
                raise Exception("server error: returned invalid json")

            return CheckApiResponse[output_as].parse_obj(obj)

    async def run_main(
        self,
        model_key: str,
        model_inputs: TModelInputs,
        api_key: str | None = None,
        output_as: type[TModelOutputs] = dict[str, Any],
    ) -> Response[TModelOutputs]:
        start_result = await self.start_api(
            model_key,
            model_inputs,
            api_key=api_key,
            start_only=False,
            output_as=output_as,
        )

        # likely we get results on first call
        if start_result.finished:
            return start_result.as_response(output_as)

        # else it's long running, so poll for result

        while True:
            result = await self.check_api(call_id=start_result.callID)
            if result.message.lower() == "success":
                return result.as_response(output_as)

    async def start_main(
        self,
        model_key: str,
        model_inputs: TModelInputs,
        api_key: str | None = None,
        output_as: type[TModelOutputs] = dict[str, Any],
    ) -> str:
        result = await self.start_api(
            model_key,
            model_inputs,
            api_key=api_key,
            start_only=True,
            output_as=output_as,
        )
        return result.callID

    async def check_main(
        self,
        api_key: str,
        call_id: str,
        output_as: type[TModelOutputs] = dict[str, Any],
    ) -> CheckApiResponse[TModelOutputs]:
        dict_out = await self.check_api(api_key, call_id, output_as=output_as)
        return dict_out

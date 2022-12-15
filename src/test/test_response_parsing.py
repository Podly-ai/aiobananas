from typing import Any, Callable, Dict, List
from aiobananas.generics import CheckApiResponse, StartApiResponse, Session, Response
from pydantic import BaseModel, validator
import pytest
import copy


class OkOutputs(BaseModel):
    test: str

    @validator("test")
    def test_must_be_test(cls, v):
        if v != "test":
            raise ValueError("test must be test")
        return v


class OkInputs(BaseModel):
    test: str


class BadOutputs:
    test: str

    def __init__(self, test: str):
        self.test = test


def test_parse_start_api_response():
    response_dict = {
        "id": "123",
        "created": 123,
        "message": "success",
        "apiVersion": "test",
        "callID": "123",
        "finished": True,
        "modelOutputs": [{"test": "test"}],
    }

    def should_throw(case: Callable[[dict], None], error=ValueError):
        case_dict = copy.deepcopy(response_dict)
        case(case_dict)
        with pytest.raises(error):
            StartApiResponse[OkOutputs].parse_obj(case_dict)

    def should_not_throw(case: Callable[[dict], None]):
        case_dict = copy.deepcopy(response_dict)
        case(case_dict)
        response = StartApiResponse[OkOutputs].parse_obj(case_dict)
        assert isinstance(response, StartApiResponse[OkOutputs])
        if response.modelOutputs is not None:
            assert isinstance(response.modelOutputs[0], OkOutputs)

    # should be able to parse the example response
    response = StartApiResponse[OkOutputs].parse_obj(response_dict)
    assert isinstance(response, StartApiResponse[OkOutputs])
    assert isinstance(response.modelOutputs[0], OkOutputs)
    assert response.modelOutputs[0].test == "test"
    assert response.finished
    assert response.callID == "123"
    assert response.id == "123"
    assert response.created == 123
    assert response.message == "success"
    assert response.apiVersion == "test"

    # should throw if any validations fail
    should_throw(lambda x: x["modelOutputs"][0].update({"test": "not_test"}))
    should_throw(lambda x: x.update({"finished": False}))
    should_throw(lambda x: x.update({"finished": True, "modelOutputs": None}))
    should_throw(lambda x: x.update({"message": "error"}), error=Exception)

    # should not throw if not finished and no model outputs
    should_not_throw(lambda x: x.update({"finished": False, "modelOutputs": None}))


def test_parse_dict_model_outputs():
    response_dict = {
        "id": "123",
        "created": 123,
        "message": "success",
        "apiVersion": "test",
        "callID": "123",
        "finished": True,
        "modelOutputs": [{"test": "test"}],
    }

    response = StartApiResponse[dict].parse_obj(response_dict)
    assert isinstance(response.modelOutputs[0], dict)
    assert response.modelOutputs[0]["test"] == "test"

    with pytest.raises(RuntimeError):
        StartApiResponse[BadOutputs].parse_obj(response_dict)


@pytest.mark.asyncio
async def test_start_api(aioresponses):
    async with Session("test-api-key") as session:
        aioresponses.post(
            "https://api.banana.dev/start/v4/",
            payload={
                "id": "123",
                "created": 123,
                "message": "success",
                "apiVersion": "test",
                "callID": "123",
                "finished": True,
                "modelOutputs": [{"test": "test"}],
            },
        )

        request = OkInputs(test="test")
        response = await session.start_api(
            "test-model-key",
            request,
            output_as=OkOutputs,
        )

        assert isinstance(response, StartApiResponse[OkOutputs])
        assert isinstance(response.modelOutputs[0], OkOutputs)
        assert response.modelOutputs[0].test == "test"

        aioresponses.post(
            "https://api.banana.dev/start/v4/",
            payload={
                "id": "123",
                "created": 123,
                "message": "success",
                "apiVersion": "test",
                "callID": "123",
                "finished": True,
                "modelOutputs": [{"test": "test"}],
            },
        )

        response = await session.start_api(
            "test-model-key",
            {"test": "test"},
            output_as=dict[str, Any],
        )

        assert isinstance(response, StartApiResponse[dict[str, Any]])
        assert isinstance(response.modelOutputs[0], dict)
        assert response.modelOutputs[0]["test"] == "test"


@pytest.mark.asyncio
async def test_check_api(aioresponses):
    async with Session("test-api-key") as session:
        aioresponses.post(
            "https://api.banana.dev/check/v4/",
            payload={
                "id": "123",
                "created": 123,
                "message": "success",
                "apiVersion": "test",
                "callID": "123",
                "modelOutputs": [{"test": "test"}],
            },
        )

        response = await session.check_api("test-call-id", output_as=OkOutputs)
        assert isinstance(response, CheckApiResponse[OkOutputs])
        assert isinstance(response.modelOutputs[0], OkOutputs)
        assert response.modelOutputs[0].test == "test"


@pytest.mark.asyncio
async def test_run_main(aioresponses):
    async with Session("test-api-key") as session:
        aioresponses.post(
            "https://api.banana.dev/start/v4/",
            payload={
                "id": "123",
                "created": 123,
                "message": "success",
                "apiVersion": "test",
                "callID": "123",
                "finished": False,
            },
        )

        aioresponses.post(
            "https://api.banana.dev/check/v4/",
            payload={
                "id": "123",
                "created": 123,
                "message": "pending",
                "apiVersion": "test",
                "callID": "123",
            },
        )

        aioresponses.post(
            "https://api.banana.dev/check/v4/",
            payload={
                "id": "123",
                "created": 123,
                "message": "success",
                "apiVersion": "test",
                "callID": "123",
                "modelOutputs": [{"test": "test"}],
            },
        )

        request = OkInputs(test="test")
        response = await session.run_main(
            "test-model-key",
            request,
            output_as=OkOutputs,
        )

        assert isinstance(response, Response[OkOutputs])
        assert isinstance(response.modelOutputs[0], OkOutputs)
        assert response.modelOutputs[0].test == "test"


def test_as_response():
    check_response = CheckApiResponse[dict[str, Any]].parse_obj(
        {
            "id": "123",
            "created": 123,
            "message": "success",
            "apiVersion": "test",
            "callID": "123",
            "modelOutputs": [{"test": "test"}],
        },
    )
    assert isinstance(check_response, CheckApiResponse[dict[str, Any]])
    assert isinstance(check_response.modelOutputs, List)
    assert isinstance(check_response.modelOutputs[0], Dict)

    response = check_response.as_response(OkOutputs)
    assert isinstance(response, Response[OkOutputs])
    assert isinstance(response.modelOutputs, list)
    assert isinstance(response.modelOutputs[0], OkOutputs)

    start_response = StartApiResponse[OkOutputs].parse_obj(
        {
            "id": "123",
            "created": 123,
            "message": "success",
            "finished": True,
            "apiVersion": "test",
            "callID": "123",
            "modelOutputs": [{"test": "test"}],
        }
    )
    assert isinstance(start_response, StartApiResponse[OkOutputs])
    assert isinstance(start_response.modelOutputs, list)
    assert isinstance(start_response.modelOutputs[0], OkOutputs)

    response = start_response.as_response(Dict[str, Any])
    assert isinstance(response, Response[Dict[str, Any]])
    assert isinstance(response.modelOutputs, List)
    assert isinstance(response.modelOutputs[0], dict)

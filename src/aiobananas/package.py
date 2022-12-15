from .generics import Session


# Generics
async def run(api_key: str, model_key: str, model_inputs: str) -> dict:
    async with Session(api_key=api_key) as session:
        out = await session.run_main(model_key=model_key, model_inputs=model_inputs)
        return out


async def start(api_key: str, model_key: str, model_inputs: str) -> dict:
    async with Session(api_key=api_key) as session:
        out = await session.start_main(model_key=model_key, model_inputs=model_inputs)
        return out


async def check(api_key: str, call_id: str) -> dict:
    async with Session(api_key=api_key) as session:
        out = await session.check_api(call_id=call_id)
        return out

"""Optional shared inference spend limit, with durable reservations for uncertain requests.

Reserve a bound on this request's prompt (estimated tokens x safety factor + fixed overhead, capped at the served
input window) plus the requested output cap before each provider request.
Successful replies replace that reservation with uncached list-price usage.
Lost replies retain their full reservation until an operator reconciles provider usage.
Requests an HTTP status error rejected before any reply body were never billed; their reservation is released.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from world_adapt.agent.context import conversation_tokens
from world_adapt.agent.errors import InfrastructureError

PROMPT_SAFETY_FACTOR = 2.0
"""Margin for the model's own tokenizer differing from the recorded estimate (`memory.TOKENIZER`)."""
PROMPT_OVERHEAD_TOKENS = 1_024
"""Fixed allowance for chat-template framing and fields the estimate does not count."""


class SpendLimitExceeded(RuntimeError):
    """Stop paid work without turning the limit into a scored model failure."""


def rejected_before_reply(error: BaseException) -> bool:
    """The provider refused the request with an HTTP status before doing any work, so nothing was billed.

    Client errors (except 408) and capacity refusals (503, 529) qualify. A 408, 500, 502 or 504 can arrive after
    the model already ran, and SDKs report an error inside an already started stream with the stream's 200 status;
    those stay unknown, like timeouts and connection errors.
    """
    status = getattr(error, "status_code", None)
    return type(status) is int and ((400 <= status < 500 and status != 408) or status in (503, 529))


def _write_ledger(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def raise_ledger_limit(path: Path, limit: float) -> None:
    """Raise the ledger cap under its file lock, creating the ledger if absent; never lowers. Idempotent."""
    import fcntl

    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("spend limit must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            data = json.loads(path.read_text())
            if data["limit_usd"] >= limit:
                return
        else:
            data = {"version": 1, "limit_usd": limit, "charged_usd": 0.0,
                    "completed_requests": 0, "reservations": {}, "rates": {}}
        data["limit_usd"] = limit
        _write_ledger(path, data)


class SpendingClient:
    def __init__(self, client, path: Path, *, limit: float, input_price: float,
                 output_price: float, input_limit: int, agent: str, cached_input_price: float | None = None):
        if any(not math.isfinite(x) or x < 0 for x in (limit, input_price, output_price)) or limit <= 0:
            raise ValueError("spend limit and token prices must be finite and nonnegative")
        if type(input_limit) is not int or input_limit < 1 or not agent:
            raise ValueError("spending requires a served input limit and agent name")
        self.client, self.path, self.limit = client, Path(path), limit
        self.input_price, self.output_price, self.input_limit = input_price, output_price, input_limit
        self.cached_input_price = input_price if cached_input_price is None else cached_input_price
        if not math.isfinite(self.cached_input_price) or not 0 <= self.cached_input_price <= input_price:
            raise ValueError("cached input price must be between zero and input price")
        self.agent = agent
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @contextmanager
    def _ledger(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text())
                # A cap raised since this process was configured (`local run --budget-usd`) governs; never a lower one.
                if data["limit_usd"] < self.limit or data["version"] != 1:
                    raise ValueError("spend ledger limit is below the configured limit, or its version differs")
            else:
                data = {"version": 1, "limit_usd": self.limit, "charged_usd": 0.0,
                        "completed_requests": 0, "reservations": {}, "rates": {}}
            if (type(data.get('charged_usd')) not in (int, float)
                    or not math.isfinite(data['charged_usd']) or data['charged_usd'] < 0
                    or not isinstance(data.get('reservations'), dict)
                    or any(type(r.get('reserved_usd')) not in (int, float)
                           or not math.isfinite(r['reserved_usd']) or r['reserved_usd'] < 0
                           for r in data['reservations'].values())):
                raise ValueError('invalid spend ledger amounts')
            rates = {"input": self.input_price, "cached_input": self.cached_input_price,
                     "output": self.output_price, "input_limit": self.input_limit}
            if self.agent in data["rates"] and data["rates"][self.agent] != rates:
                raise ValueError("spend ledger agent rate or context limit differs")
            data["rates"][self.agent] = rates
            yield data
            _write_ledger(self.path, data)

    def create(self, **kwargs):
        output_limit = kwargs.get("max_completion_tokens")
        if type(output_limit) is not int or output_limit < 1:
            raise ValueError("spending requires an explicit positive reply token cap")
        estimate = conversation_tokens(kwargs.get("messages") or [], kwargs.get("tools"))
        prompt_bound = min(self.input_limit, math.ceil(estimate * PROMPT_SAFETY_FACTOR) + PROMPT_OVERHEAD_TOKENS)
        reservation = (prompt_bound * self.input_price + output_limit * self.output_price) / 1_000_000
        request_id = uuid.uuid4().hex
        with self._ledger() as data:
            outstanding = sum(r["reserved_usd"] for r in data["reservations"].values())
            if data["charged_usd"] + outstanding + reservation > data["limit_usd"]:
                raise SpendLimitExceeded("inference spend limit reached; preserve and resume after reconciliation")
            data["reservations"][request_id] = {"agent": self.agent, "reserved_usd": reservation,
                                                "reserved_prompt_tokens": prompt_bound, "status": "in_flight"}
        try:
            reply = self.client.chat.completions.create(**kwargs)
            usage = getattr(reply, "usage", None)
            prompt, completion = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            if (type(prompt) is not int or type(completion) is not int or prompt < 0 or completion < 0
                    or prompt > self.input_limit or completion > output_limit or type(cached) is not int
                    or not 0 <= cached <= prompt):
                raise InfrastructureError("missing or out-of-range usage; full request reservation retained")
        except BaseException as error:
            with self._ledger() as data:
                if rejected_before_reply(error):
                    del data["reservations"][request_id]
                    data["rejected_requests"] = data.get("rejected_requests", 0) + 1
                else:
                    data["reservations"][request_id]["status"] = "unknown"
            raise
        with self._ledger() as data:
            del data["reservations"][request_id]
            charge = ((prompt - cached) * self.input_price + cached * self.cached_input_price
                      + completion * self.output_price) / 1_000_000
            data["charged_usd"] += charge
            data["completed_requests"] += 1
            agent_usage = data.setdefault('agents', {}).setdefault(self.agent, {
                'charged_usd': 0.0, 'prompt_tokens': 0, 'cached_tokens': 0, 'completion_tokens': 0, 'requests': 0})
            for key, value in [('charged_usd', charge), ('prompt_tokens', prompt), ('cached_tokens', cached),
                               ('completion_tokens', completion), ('requests', 1)]:
                agent_usage[key] += value
        return reply


def wrap_client(client):
    path = os.environ.get("WORLD_ADAPT_SPEND_LEDGER")
    if not path:
        if any(k in os.environ for k in ('WORLD_ADAPT_SPEND_LIMIT_USD', 'WORLD_ADAPT_SPEND_AGENT')):
            raise ValueError('spend settings require a nonempty WORLD_ADAPT_SPEND_LEDGER')
        return client
    return SpendingClient(client, Path(path), limit=float(os.environ["WORLD_ADAPT_SPEND_LIMIT_USD"]),
                          input_price=float(os.environ["WORLD_ADAPT_INPUT_PRICE"]),
                          output_price=float(os.environ["WORLD_ADAPT_OUTPUT_PRICE"]),
                          input_limit=int(os.environ["WORLD_ADAPT_INPUT_LIMIT"]),
                          cached_input_price=float(os.environ.get("WORLD_ADAPT_CACHED_INPUT_PRICE", os.environ["WORLD_ADAPT_INPUT_PRICE"])),
                          agent=os.environ["WORLD_ADAPT_SPEND_AGENT"])

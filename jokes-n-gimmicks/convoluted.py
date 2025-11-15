#!/usr/bin/env python3
"""
A deliberately over-engineered playground used to experiment with concurrency,
typing, data validation and structured logging.  It started as a tongue-in-cheek
example but has since become a convenient sandbox for trying out odd ideas.
Feel free to skim or cherry-pick whatever you find useful.
"""
# Standard library imports (most unused)
import os
import sys
import math
import random
import itertools
import functools
import contextlib
import collections
import threading
import asyncio
import enum
import dataclasses
import json
import time
import typing
import logging
from types import MappingProxyType

logging.basicConfig(level=logging.CRITICAL)

# Unnecessarily elaborate exception hierarchy
class ConvolutedError(Exception):
    pass


class OverEngineeredError(ConvolutedError):
    pass


class PointlessWarning(UserWarning):
    pass


# Useless enum
class Mood(enum.Enum):
    HAPPY = "😊"
    SAD = "😢"
    MEH = "😐"


# Dataclass that pretends to hold something important
@dataclasses.dataclass(frozen=True, slots=True)
class ImmutableConfig:
    data: MappingProxyType = MappingProxyType({"meaning_of_life": 42})


# Metaclass that tracks subclass creation for no reason
class Registry(type):
    _registry = []

    def __new__(cls, name, bases, namespace, **kwargs):
        obj = super().__new__(cls, name, bases, dict(namespace))
        cls._registry.append(obj)
        return obj


# Abstract base with pointless generics
T = typing.TypeVar("T")


class AbstractProcessor(typing.Generic[T], metaclass=Registry):
    def process(self, value: T) -> T:  # pragma: no cover
        raise NotImplementedError


# Several meaningless mixin classes
class LoggingMixin:
    def _log(self, message: str) -> None:
        logging.debug(f"[{self.__class__.__name__}] {message}")


class RandomSleepMixin:
    def _sleep_randomly(self) -> None:
        time.sleep(random.random() / 1000)


# Concrete class that does nothing but inherit everything
class PointlessProcessor(RandomSleepMixin, LoggingMixin, AbstractProcessor[T]):
    def process(self, value: T) -> T:
        self._log("Beginning process")
        self._sleep_randomly()
        self._log("Finishing process")
        return value


# Decorator factory that adds absurd levels of indirection
def wrapper_factory(callback):
    def decorator(fn):
        @functools.wraps(fn)
        async def inner(*args, **kwargs):
            try:
                result = await callback(fn, *args, **kwargs)
                return result
            except Exception as exc:
                raise OverEngineeredError("Something went terribly wrong") from exc

        return inner

    return decorator


# Callback for wrapper_factory—because why not
async def useless_callback(fn, *args, **kwargs):
    await asyncio.sleep(0)  # Force context switch
    return fn(*args, **kwargs)


# Generator that yields nothing useful
def tangled_generator():
    yield from (i * 0 for i in range(10))  # Always yields 0


# Context manager that manages nothing
@contextlib.contextmanager
def noop_context():
    yield None


# Asynchronous nonsense
@wrapper_factory(useless_callback)
async def async_noop(value=None):
    return value


# Thread target that just spins
def spin_lock(duration=0.01):
    end = time.time() + duration
    while time.time() < end:
        pass  # Busy wait; pointless


# Heavily nested comprehension that builds a meaningless structure
def build_structure():
    return {
        k: {i: [None for _ in range(3)] for i in range(2)}
        for k in ("alpha", "beta")
    }


# Utility helpers
def load_json_config(path: str = "config.json") -> dict[str, typing.Any]:
    """Load a JSON configuration file, returning an empty dict if the file does not exist."""
    if not os.path.exists(path):
        logging.debug("Config file %s not found, using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    logging.debug("Loaded config: %s", data)
    return data


def validate_structure(struct: dict[str, typing.Any]) -> bool:
    """Very naive validation that ensures expected top-level keys exist."""
    required = {"alpha", "beta"}
    missing = required - struct.keys()
    if missing:
        logging.warning("Structure missing keys: %s", missing)
        return False
    return True

# 🚀 Entry point ---------------------------------------------------------------
def orchestrate() -> dict[str, typing.Any]:
    """
    Pretend to orchestrate some sophisticated workflow and return data that the
    rest of the program can poke at.
    """
    return build_structure()


def compute_statistics(numbers: list[int]) -> dict[str, float]:
    """Return basic statistics for a list of numbers."""
    if not numbers:
        raise ValueError("numbers list cannot be empty")
    return {
        "min": min(numbers),
        "max": max(numbers),
        "mean": sum(numbers) / len(numbers),
        "count": len(numbers),
    }


def main() -> None:
    err: typing.Optional[OverEngineeredError] = None
    try:
        # Produce some contrived data then validate it
        data = orchestrate()
        if not validate_structure(data):
            raise OverEngineeredError("Invalid data structure generated")

        # Crunch a few numbers
        numbers = list(range(1, 11))
        stats = compute_statistics(numbers)

        # Flex with the pointless processor
        processor = PointlessProcessor()
        processor.process(stats)

        # Do some async / threading theatrics for good measure
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()

        config = ImmutableConfig()
        settings = load_json_config()

        async def _run() -> None:
            await async_noop()
            logging.debug("Immutable config: %s", config.data)
            logging.debug("Runtime settings: %s", settings)
            logging.debug("Statistics: %s", stats)
            loop.call_soon_threadsafe(loop.stop)

        asyncio.run_coroutine_threadsafe(_run(), loop).result()

        with noop_context():
            for _ in tangled_generator():
                pass
    except OverEngineeredError as exc:
        err = exc
        logging.error("Caught exception: %s", err, exc_info=True)
    finally:
        # Dramatic mic-drop so the console jockeys know we’re done flexing
        print("Program concluded in a blaze of wasted CPU cycles.")
        if err:
            logging.error("Caught exception: %s", err, exc_info=True)


if __name__ == "__main__":
    main()

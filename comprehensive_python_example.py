# Comprehensive Python Example
# Showcases Python features with concise comments for learning and reference.

import os
import sys
import time
import json
import asyncio
import threading
import multiprocessing
from datetime import datetime
from collections import defaultdict, Counter
from functools import wraps, partial
from itertools import chain, islice
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable
import sqlite3
import socket
import requests

# Basic Syntax and Data Types
def basic_syntax_demo():
    x, y, z = 42, 3.14, 3 + 4j  # Numeric types
    print(f"Numeric: {x}, {y}, {z}")
    print(f"Conversions: {float(x)}, {int(y)}")
    text = "Hello, Python!"  # Strings and bytes
    print(f"String, bytes: {text}, {text.encode('utf-8')}")
    collections = ([1, 2, 3], (4, 5, 6), {7, 8, 9}, {"a": 1, "b": 2})  # Collections
    print(f"Collections: {collections}")

# Control Structures
def control_structures_demo():
    # Nested loops with break
    for i in range(10):
        for j in range(2):
            if i == 5 and j == 1:
                print("Breaking loop")
                break
        else:
            continue
        break
    # Match statement (Python 3.10+)
    value = 42
    match value:
        case 42:
            print("The answer")
        case _:
            print("Other number")

# Functions
def function_demo(x: int, y: int = 10, *args, **kwargs) -> int:
    def nested(z: int) -> int:  # Nested function
        return z * 2
    total = sum(args) + x + y + kwargs.get("extra", 0)
    return nested(total)

# Exception Handling
class CustomError(Exception):
    pass

def exception_handling_demo():
    try:
        raise CustomError("Error!")
    except CustomError as e:
        print(f"Caught: {e}")
    finally:
        print("Cleanup")

# File I/O
def file_io_demo():
    # Text file
    with open("example.txt", "w") as f:
        f.write("Hello, File!")
    with open("example.txt", "r") as f:
        print(f"Text: {f.read()}")
    # JSON file
    data = {"key": "value"}
    with open("example.json", "w") as f:
        json.dump(data, f)
    with open("example.json", "r") as f:
        print(f"JSON: {json.load(f)}")

# Object-Oriented Programming
class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    def __init__(self, name: str):
        self._name = name
    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, value: str):
        self._name = value
    def speak(self) -> str:
        return "Woof!"

class Mammal:
    def walk(self) -> str:
        return "Walking"

class Puppy(Dog, Mammal):
    pass

# Advanced Features
def generator_demo():
    def fibonacci(n: int):  # Generator
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b
    print(f"Fibonacci: {list(fibonacci(5))}")
    print(f"Gen expr: {list(x**2 for x in range(5))}")

def decorator_demo():
    def timer(func: Callable) -> Callable:  # Decorator
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {time.time() - start:.2f}s")
            return result
        return wrapper
    @timer
    def slow_function():
        time.sleep(1)
        return "Done"
    print(slow_function())

class MyContext:
    def __enter__(self):
        print("Entering")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting")

# Asynchronous Programming
async def async_demo():
    await asyncio.sleep(1)
    print("Async done")

# Standard Library
def standard_library_demo():
    print(f"Time: {datetime.now()}")
    print(f"Counter: {Counter('abracadabra')}")
    d = defaultdict(int)
    d["key"] += 1
    print(f"Defaultdict: {d}")

# Concurrency and Parallelism
def threading_demo():
    def worker():
        print(f"Thread {threading.current_thread().name}")
    t = threading.Thread(target=worker)
    t.start()
    t.join()

# Global function for multiprocessing
def square(n: int):
    print(f"Process {multiprocessing.current_process().name} squaring {n}: {n*n}")

def multiprocessing_demo():
    p = multiprocessing.Process(target=square, args=(5,))
    p.start()
    p.join()

# Networking
def socket_demo():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 9999))
    server.listen(1)
    print("Listening...")
    conn, addr = server.accept()
    data = conn.recv(1024)
    conn.send(b"Received: " + data)
    conn.close()
    server.close()

def requests_demo():
    try:
        r = requests.get("https://api.github.com")
        print(f"Status: {r.status_code}")
    except requests.RequestException as e:
        print(f"Failed: {e}")

# Database
def sqlite_demo():
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    c.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()
    print(f"DB: {c.execute('SELECT * FROM users').fetchone()}")
    conn.close()

# Testing
def testing_demo():
    def add(a: int, b: int) -> int:
        return a + b
    assert add(2, 3) == 5, "Failed"
    print("Tests passed")

# Performance
def performance_demo():
    import timeit
    print(f"List comp: {timeit.timeit('[x**2 for x in range(1000)]', number=1000):.4f}s")
    print(f"Loop: {timeit.timeit('squares = []\nfor x in range(1000):\n    squares.append(x**2)', number=1000):.4f}s")

# Main
if __name__ == "__main__":
    basic_syntax_demo()
    control_structures_demo()
    print(f"Function: {function_demo(1, 2, 3, 4, extra=5)}")
    exception_handling_demo()
    file_io_demo()
    puppy = Puppy("Max")
    print(f"Puppy: {puppy.speak()}, {puppy.walk()}")
    generator_demo()
    decorator_demo()
    with MyContext():
        print("In context")
    asyncio.run(async_demo())
    standard_library_demo()
    threading_demo()
    multiprocessing_demo()
    # Uncomment for additional demos
    # socket_demo()  # Needs client
    # requests_demo()
    # sqlite_demo()
    testing_demo()
    performance_demo()
    print("Demo complete!")
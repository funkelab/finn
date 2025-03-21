import inspect
import time
import types
from unittest.mock import patch

import pytest
from app_model.types import KeyBinding, KeyCode, KeyMod

from finn.utils import key_bindings
from finn.utils.key_bindings import (
    KeymapHandler,
    KeymapProvider,
    _bind_keymap,
    _bind_user_key,
    _get_user_keymap,
    bind_key,
)


@pytest.mark.key_bindings
def test_bind_key():
    kb = {}

    # bind
    def forty_two():
        return 42

    bind_key(kb, "A", forty_two)
    assert kb == {KeyBinding.from_str("A"): forty_two}

    # overwrite
    def spam():
        return "SPAM"

    with pytest.raises(ValueError, match="already used"):
        bind_key(kb, "A", spam)

    bind_key(kb, "A", spam, overwrite=True)
    assert kb == {KeyBinding.from_str("A"): spam}

    # unbind
    bind_key(kb, "A", None)
    assert kb == {}

    # check signature
    # blocker
    bind_key(kb, "A", ...)
    assert kb == {KeyBinding.from_str("A"): ...}

    # catch-all
    bind_key(kb, ..., ...)
    assert kb == {KeyBinding.from_str("A"): ..., ...: ...}

    # typecheck
    with pytest.raises(TypeError):
        bind_key(kb, "B", "not a callable")

    # app-model representation
    kb = {}
    bind_key(kb, KeyMod.Shift | KeyCode.KeyA, ...)
    (key,) = kb.keys()
    assert key == KeyBinding.from_str("Shift-A")


@pytest.mark.key_bindings
def test_bind_key_decorator():
    kb = {}

    @bind_key(kb, "A")
    def foo(): ...

    assert kb == {KeyBinding.from_str("A"): foo}


@pytest.mark.key_bindings
def test_keymap_provider():
    class Foo(KeymapProvider): ...

    assert Foo.class_keymap == {}

    foo = Foo()
    assert foo.keymap == {}

    class Bar(Foo): ...

    assert Bar.class_keymap == {}
    assert Bar.class_keymap is not Foo.class_keymap

    class Baz(KeymapProvider):
        class_keymap = {"A": ...}

    assert Baz.class_keymap == {KeyBinding.from_str("A"): ...}


@pytest.mark.key_bindings
def test_bind_keymap():
    class Foo: ...

    def bar(foo):
        return foo

    def baz(foo):
        return foo

    keymap = {"A": bar, "B": baz, "C": ...}

    foo = Foo()

    assert _bind_keymap(keymap, foo) == {
        "A": types.MethodType(bar, foo),
        "B": types.MethodType(baz, foo),
        "C": ...,
    }


class Foo(KeymapProvider):
    class_keymap = {
        "A": lambda x: setattr(x, "A", ...),
        "B": lambda x: setattr(x, "B", ...),
        "C": lambda x: setattr(x, "C", ...),
        "D": ...,
    }

    def __init__(self) -> None:
        self.keymap = {
            "B": lambda x: setattr(x, "B", None),  # overwrite
            "E": lambda x: setattr(x, "E", None),  # new entry
            "C": ...,  # blocker
        }


class Bar(KeymapProvider):
    class_keymap = {"E": lambda x: setattr(x, "E", 42)}


class Baz(Bar):
    class_keymap = {"F": lambda x: setattr(x, "F", 16)}


@pytest.mark.key_bindings
def test_handle_single_keymap_provider():
    foo = Foo()

    handler = KeymapHandler()
    handler.keymap_providers = [foo]

    assert handler.keymap_chain.maps == [
        _get_user_keymap(),
        _bind_keymap(foo.keymap, foo),
        _bind_keymap(foo.class_keymap, foo),
    ]
    assert handler.active_keymap == {
        KeyBinding.from_str("A"): types.MethodType(
            foo.class_keymap[KeyBinding.from_str("A")], foo
        ),
        KeyBinding.from_str("B"): types.MethodType(
            foo.keymap[KeyBinding.from_str("B")], foo
        ),
        KeyBinding.from_str("E"): types.MethodType(
            foo.keymap[KeyBinding.from_str("E")], foo
        ),
    }

    # non-overwritten class keybinding
    # 'A' in Foo and not foo
    assert not hasattr(foo, "A")
    handler.press_key("A")
    assert foo.A is ...

    # keybinding blocker on class
    # 'D' in Foo and not foo but has no func
    handler.press_key("D")
    assert not hasattr(foo, "D")

    # non-overwriting instance keybinding
    # 'E' not in Foo and in foo
    assert not hasattr(foo, "E")
    handler.press_key("E")
    assert foo.E is None

    # overwriting instance keybinding
    # 'B' in Foo and in foo; foo has priority
    assert not hasattr(foo, "B")
    handler.press_key("B")
    assert foo.B is None

    # keybinding blocker on instance
    # 'C' in Foo and in Foo; foo has priority but no func
    handler.press_key("C")
    assert not hasattr(foo, "C")


@pytest.mark.key_bindings
@patch("finn.utils.key_bindings.USER_KEYMAP", new_callable=dict)
def test_bind_user_key(keymap_mock):
    foo = Foo()
    bar = Bar()
    handler = KeymapHandler()
    handler.keymap_providers = [bar, foo]

    x = 0

    @_bind_user_key("D")
    def abc():
        nonlocal x
        x = 42

    assert handler.active_keymap == {
        KeyBinding.from_str("A"): types.MethodType(
            foo.class_keymap[KeyBinding.from_str("A")], foo
        ),
        KeyBinding.from_str("B"): types.MethodType(
            foo.keymap[KeyBinding.from_str("B")], foo
        ),
        KeyBinding.from_str("D"): abc,
        KeyBinding.from_str("E"): types.MethodType(
            bar.class_keymap[KeyBinding.from_str("E")], bar
        ),
    }

    handler.press_key("D")

    assert x == 42


@pytest.mark.key_bindings
def test_handle_multiple_keymap_providers():
    foo = Foo()
    bar = Bar()
    handler = KeymapHandler()
    handler.keymap_providers = [bar, foo]

    assert handler.keymap_chain.maps == [
        _get_user_keymap(),
        _bind_keymap(bar.keymap, bar),
        _bind_keymap(bar.class_keymap, bar),
        _bind_keymap(foo.keymap, foo),
        _bind_keymap(foo.class_keymap, foo),
    ]
    assert handler.active_keymap == {
        KeyBinding.from_str("A"): types.MethodType(
            foo.class_keymap[KeyBinding.from_str("A")], foo
        ),
        KeyBinding.from_str("B"): types.MethodType(
            foo.keymap[KeyBinding.from_str("B")], foo
        ),
        KeyBinding.from_str("E"): types.MethodType(
            bar.class_keymap[KeyBinding.from_str("E")], bar
        ),
    }

    # check 'bar' callback
    # 'E' in bar and foo; bar takes priority
    assert not hasattr(bar, "E")
    handler.press_key("E")
    assert bar.E == 42

    # check 'foo' callback
    # 'B' not in bar and in foo
    handler.press_key("B")
    assert not hasattr(bar, "B")

    # catch-all key combo
    # if key not found in 'bar' keymap; default to this binding
    def catch_all(x):
        x.catch_all = True

    bar.class_keymap[...] = catch_all
    assert handler.active_keymap == {
        ...: types.MethodType(catch_all, bar),
        KeyBinding.from_str("E"): types.MethodType(
            bar.class_keymap[KeyBinding.from_str("E")], bar
        ),
    }
    assert not hasattr(bar, "catch_all")
    handler.press_key("Z")
    assert bar.catch_all is True

    # empty
    bar.class_keymap[...] = ...
    assert handler.active_keymap == {
        KeyBinding.from_str("E"): types.MethodType(
            bar.class_keymap[KeyBinding.from_str("E")], bar
        ),
    }
    del foo.B
    handler.press_key("B")
    assert not hasattr(foo, "B")


@pytest.mark.key_bindings
def test_inherited_keymap():
    baz = Baz()
    handler = KeymapHandler()
    handler.keymap_providers = [baz]

    assert handler.keymap_chain.maps == [
        _get_user_keymap(),
        _bind_keymap(baz.keymap, baz),
        _bind_keymap(baz.class_keymap, baz),
        _bind_keymap(Bar.class_keymap, baz),
    ]
    assert handler.active_keymap == {
        KeyBinding.from_str("F"): types.MethodType(
            baz.class_keymap[KeyBinding.from_str("F")], baz
        ),
        KeyBinding.from_str("E"): types.MethodType(
            Bar.class_keymap[KeyBinding.from_str("E")], baz
        ),
    }


@pytest.mark.key_bindings
def test_handle_on_release_bindings():
    def make_42(x):
        # on press
        x.SPAM = 42
        if False:
            yield
            # on release
            # do nothing, but this will make it a generator function

    def add_then_subtract(x):
        # on press
        x.aliiiens += 3
        yield
        # on release
        x.aliiiens -= 3

    class Baz(KeymapProvider):
        aliiiens = 0
        class_keymap = {
            KeyCode.Shift: make_42,
            "Control-Shift-B": add_then_subtract,
        }

    baz = Baz()
    handler = KeymapHandler()
    handler.keymap_providers = [baz]

    # one-statement generator function
    assert not hasattr(baz, "SPAM")
    handler.press_key("Shift")
    assert baz.SPAM == 42

    # two-statement generator function
    assert baz.aliiiens == 0
    handler.press_key("Control-Shift-B")
    assert baz.aliiiens == 3
    handler.release_key("Control-Shift-B")
    assert baz.aliiiens == 0

    # order of modifiers should not matter
    handler.press_key("Shift-Control-B")
    assert baz.aliiiens == 3
    handler.release_key("B")
    assert baz.aliiiens == 0


@pytest.mark.key_bindings
def test_bind_key_method():
    class Foo2(KeymapProvider): ...

    foo = Foo2()

    # instance binding
    foo.bind_key("A", lambda: 42)
    assert foo.keymap[KeyBinding.from_str("A")]() == 42

    # class binding
    @Foo2.bind_key("B")
    def bar():
        return "SPAM"

    assert Foo2.class_keymap[KeyBinding.from_str("B")] is bar


@pytest.mark.key_bindings
def test_bind_key_doc():
    doc = inspect.getdoc(bind_key)
    doc = doc.split("Notes\n-----\n")[-1]

    assert doc == inspect.getdoc(key_bindings)


def test_key_release_callback(monkeypatch):
    called = False
    called2 = False
    monkeypatch.setattr(time, "time", lambda: 1)

    class Foo(KeymapProvider): ...

    foo = Foo()

    handler = KeymapHandler()
    handler.keymap_providers = [foo]

    def _call():
        nonlocal called2
        called2 = True

    @Foo.bind_key("K")
    def callback(x):
        nonlocal called
        called = True
        return _call

    handler.press_key("K")
    assert called
    assert not called2
    handler.release_key("K")
    assert not called2

    handler.press_key("K")
    assert called
    assert not called2
    monkeypatch.setattr(time, "time", lambda: 2)
    handler.release_key("K")
    assert called2

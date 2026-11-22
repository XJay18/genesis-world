"""
Arrays a Quadrants kernel may write and may read, from its source.

The backward pass of a recorded substep saves the values a launch overwrites that an earlier launch reads (see
LaunchRecorder). Quadrants records the struct members a compiled kernel uses, but not which of them it writes or reads,
so the writes and the reads are found in the Python source of the kernel and of the funcs it calls. The analysis
over-approximates: both branches of a static condition count, so a write or a read it reports may never happen, which
costs a copy, never a gradient. A func the source does not name escapes it: the paths of what such a func is passed
are reported apart, and the recorder takes a launch passing it one of its arrays as reading every array it uses.
"""

import ast
import builtins
import inspect
import textwrap
from collections.abc import Callable
from typing import NamedTuple

from quadrants.lang._quadrants_callable import QuadrantsCallable
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import Matrix

from genesis.utils.cache import OwnedCache


class Callee(NamedTuple):
    """A func a call reaches: its Python function, the object its first parameter is bound to and the path of it."""

    fn: Callable
    owner: object | None
    owner_path: tuple[str, ...]


class AccessedPaths(NamedTuple):
    """The paths of the parameters of a kernel or a func that it may write, and those it may read."""

    written: frozenset[tuple[str, ...]]
    read: frozenset[tuple[str, ...]]
    unresolved: frozenset[tuple[str, ...]]
    """The paths passed to a func the source does not name, under which it may access any array unseen."""


# The paths each kernel or func may write and read, by function and bound object, each entry living as long as the bound
# object (the function itself for a free function)
_ACCESSED_PATHS_CACHE = OwnedCache()

# The atomic operations whose adjoint takes no value of the array they add to, which they write without reading it
_ATOMIC_LINEAR = ("atomic_add", "atomic_sub")

# The methods of the vectors, the matrices and the expressions of Quadrants, which name no func
_VALUE_METHODS = frozenset(
    [name for value_type in (Matrix, Expr) for name, value in vars(value_type).items() if callable(value)]
)


# FIXME: quadrants#1001 - Quadrants exposes no write or read set of a kernel, so they are derived from the source here,
# which duplicates the analysis of its frontend and counts both branches of a static condition.
def get_accessed_paths(fn: Callable, owner: object | None = None) -> AccessedPaths:
    """
    Return the paths of the parameters of a kernel or a func that it may write, and those it may read.

    A path is the name of a parameter followed by the struct fields reaching the array, such as
    `("dyn_state", "dofs", "vel")`. A write is an assignment to an element, an augmented assignment, the first argument
    of an atomic operation or a `fill`, in the function itself or in a func it calls with the array as argument. A read
    is the load of an element, the target of an augmented assignment or the first argument of an atomic operation other
    than an addition or a subtraction, whose adjoint takes no value of the array. `owner` is the object the first
    parameter of a method is bound to, through which it reaches the funcs of other objects. A func the source does not
    name, held by a parameter or by an element of an array or returned by a call, such as `materials[i].update(state, i)`
    or `qd.static(f)(state, i)`, may access anything under the paths of the arrays and structs it is passed and of the
    object holding it, which the paths `unresolved` report.
    """
    cache_key = (fn, id(owner))
    paths_cached = _ACCESSED_PATHS_CACHE.find(cache_key)
    if paths_cached is not None:
        return paths_cached
    func_def = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    params = [arg.arg for arg in (*func_def.args.posonlyargs, *func_def.args.args, *func_def.args.kwonlyargs)]
    owner_name = params[0] if owner is not None else None
    aliases = {param: {(param,)} for param in params}
    # The same names bound to whole arrays and structs, without the elements loaded from them
    aliases_whole = {param: {(param,)} for param in params}
    # Local names bound to a parameter or to fields of one, until no binding adds a path
    assigns = [node for node in ast.walk(func_def) if isinstance(node, ast.Assign)]
    is_growing = True
    while is_growing:
        is_growing = False
        for node in assigns:
            for target in node.targets:
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    bindings = zip(target.elts, node.value.elts)
                else:
                    bindings = ((target, node.value),)
                for name, value in bindings:
                    for names_paths, is_element in ((aliases, True), (aliases_whole, False)):
                        paths = _resolve(value, names_paths, is_element)
                        if isinstance(name, ast.Name) and paths and not paths <= names_paths.get(name.id, set()):
                            names_paths.setdefault(name.id, set()).update(paths)
                            is_growing = True

    # Local names bound to a func, directly or through 'qd.static'
    funcs = {}
    for node in assigns:
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "static":
            value = value.args[0]
        callee = _find_callee(value, fn, {}, owner_name, owner)
        for target in node.targets:
            if isinstance(target, ast.Name) and callee is not None:
                funcs[target.id] = callee

    paths_written, paths_read, paths_unresolved = set(), set(), set()
    # The element loads that are the first argument of a linear atomic operation, which writes them alone
    loads_atomic = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Subscript):
            if (isinstance(node.func, ast.Attribute) and node.func.attr in _ATOMIC_LINEAR) or (
                isinstance(node.func, ast.Name) and node.func.id in _ATOMIC_LINEAR
            ):
                loads_atomic.add(id(node.args[0]))
    for node in ast.walk(func_def):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and id(node) not in loads_atomic:
            paths_read |= _resolve(node.value, aliases)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                paths_written |= _resolve_store(target, aliases)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            paths_written |= _resolve_store(node.target, aliases)
            if isinstance(node, ast.AugAssign):
                paths_read |= _resolve_store(node.target, aliases)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                name = ""
            if name.startswith("atomic_") and node.args:
                paths_written |= _resolve(node.args[0], aliases)
            elif name == "fill" and isinstance(node.func, ast.Attribute):
                paths_written |= _resolve(node.func.value, aliases)
            else:
                callee = _find_callee(node.func, fn, funcs, owner_name, owner)
                if callee is None:
                    if _may_name_func(node.func, fn, aliases, owner_name, owner):
                        # The func reaches the object holding it and what it is passed, an element passed being a
                        # value whose load counts above
                        holder = node.func.value if isinstance(node.func, ast.Attribute) else node.func
                        for expr in (holder, *node.args, *[keyword.value for keyword in node.keywords]):
                            paths_unresolved |= _resolve(expr, aliases_whole, is_element=False)
                    continue
                # The arrays the func accesses through its parameters, bound to the arguments of the call and, for a
                # method, to the object it is bound to, the paths it passes on unresolved to whole arrays and structs
                callee_params = list(inspect.signature(callee.fn).parameters)
                owner_bindings = [] if callee.owner is None else [(callee_params.pop(0), {callee.owner_path})]
                params_args = [
                    *zip(callee_params, node.args),
                    *[(keyword.arg, keyword.value) for keyword in node.keywords if keyword.arg is not None],
                ]
                bindings = [*owner_bindings, *[(param, _resolve(arg, aliases)) for param, arg in params_args]]
                bindings_whole = [
                    *owner_bindings,
                    *[(param, _resolve(arg, aliases_whole, is_element=False)) for param, arg in params_args],
                ]
                callee_paths = get_accessed_paths(callee.fn, callee.owner)
                for callee_paths_accessed, paths_accessed, args_bindings in (
                    (callee_paths.written, paths_written, bindings),
                    (callee_paths.read, paths_read, bindings),
                    (callee_paths.unresolved, paths_unresolved, bindings_whole),
                ):
                    for callee_param, arg_paths in args_bindings:
                        for callee_path in callee_paths_accessed:
                            if callee_path[0] == callee_param:
                                paths_accessed |= {(*path, *callee_path[1:]) for path in arg_paths}
    params_paths = AccessedPaths(
        written=frozenset([path for path in paths_written if path[0] in params]),
        read=frozenset([path for path in paths_read if path[0] in params]),
        unresolved=frozenset([path for path in paths_unresolved if path[0] in params]),
    )
    return _ACCESSED_PATHS_CACHE.own(fn if owner is None else owner, cache_key, params_paths)


def _resolve(node: ast.expr, aliases: dict[str, set[tuple[str, ...]]], is_element: bool = True) -> set[tuple[str, ...]]:
    """
    Return the paths an expression may designate: a name bound to a path, its fields, and elements of them.

    The elements count where `is_element` holds, an element standing for the array holding it.
    """
    if isinstance(node, ast.Name):
        return set(aliases.get(node.id, ()))
    if isinstance(node, ast.Attribute):
        return {(*path, node.attr) for path in _resolve(node.value, aliases, is_element)}
    if isinstance(node, ast.Subscript) and is_element:
        return _resolve(node.value, aliases)
    return set()


def _resolve_store(target: ast.expr, aliases: dict[str, set[tuple[str, ...]]]) -> set[tuple[str, ...]]:
    """Return the paths of the arrays an assignment target writes, the element of an array or a field of one."""
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*[_resolve_store(elt, aliases) for elt in target.elts])
    if isinstance(target, ast.Subscript):
        return _resolve(target.value, aliases)
    if isinstance(target, ast.Attribute) and any([isinstance(node, ast.Subscript) for node in ast.walk(target.value)]):
        return _resolve(target.value, aliases)
    return set()


def _find_callee(
    node: ast.expr, fn: Callable, funcs: dict[str, Callee], owner_name: str | None, owner: object | None
) -> Callee | None:
    """
    Return the Quadrants func an expression names, None for anything else.

    `fn` is the function whose source holds the expression, `funcs` the local names it binds to a func, and `owner` the
    object its first parameter, named `owner_name`, is bound to.
    """
    if isinstance(node, ast.Name) and node.id in funcs:
        return funcs[node.id]
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    names.reverse()
    if node.id == owner_name:
        # A method of the owner or of an object it holds, bound to that object
        callee_owner = owner
        for name in names[:-1]:
            callee_owner = getattr(callee_owner, name, None)
        value = getattr(type(callee_owner), names[-1], None) if names else None
        owner_path = (owner_name, *names[:-1])
        if value is None and names:
            # A func the object holds as an attribute of its own, unbound
            value, callee_owner, owner_path = getattr(callee_owner, names[-1], None), None, ()
    else:
        closure = zip(fn.__code__.co_freevars, [cell.cell_contents for cell in fn.__closure__ or ()])
        scope = {**fn.__globals__, **dict(closure)}
        if node.id not in scope:
            return None
        value = scope[node.id]
        for name in names:
            value = getattr(value, name, None)
        callee_owner, owner_path = None, ()
    if not isinstance(value, QuadrantsCallable) or value.fn is fn:
        return None
    return Callee(value.fn, callee_owner, owner_path)


def _may_name_func(
    node: ast.expr, fn: Callable, aliases: dict[str, set[tuple[str, ...]]], owner_name: str | None, owner: object | None
) -> bool:
    """
    Return whether the callee of a call that '_find_callee' does not resolve may be a func all the same.

    A method of a vector, a matrix or an expression names none, nor does a method of a local value, nor a value that the
    owner or the scope of `fn` holds (a builtin, a type, a Python function). A parameter of `fn`, or a local bound to
    one, may hold a func (a template argument, a material), and so may an element of an array, the result of a call, a
    local called directly and an attribute the owner lacks.
    """
    if isinstance(node, ast.Attribute) and node.attr in _VALUE_METHODS:
        return False
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return True
    closure = zip(fn.__code__.co_freevars, [cell.cell_contents for cell in fn.__closure__ or ()])
    scope = {**vars(builtins), **fn.__globals__, **dict(closure)}
    if node.id == owner_name:
        value = owner
    elif node.id in aliases:
        return True
    elif node.id in scope:
        value = scope[node.id]
    else:
        return not names
    for name in reversed(names):
        value = getattr(value, name, None)
    return value is None

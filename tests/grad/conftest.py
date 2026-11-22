import dataclasses
import operator
import weakref
import xml.etree.ElementTree as ET

import pytest

from ..conftest import lazy_imports

# Imported on first use (see pytest_sessionstart in tests/conftest.py)
with lazy_imports():
    import torch

    import quadrants as qd

    import genesis as gs
    from genesis.grad.bridge import LaunchRecorder
    from genesis.utils.misc import qd_from_torch, qd_to_torch, qd_zerocopy_view


@pytest.fixture(autouse=True)
def step_overwrites(backend, monkeypatch):
    # Check that the backward pass of a recorded rollout reverses every entry at the values it left (see LaunchRecorder),
    # without the source analysis the recorder saves by: each reversal runs twice from the same arrays and adjoints,
    # first restoring after each entry every array it changed that an earlier entry used (upstream's rule, over the
    # writes the forward pass showed), then as the recorder does, and the gradients of both must be bitwise equal. The
    # reversals of a backend repeat bit for bit on CPU alone, where it runs.
    if isinstance(backend, str):
        backend = getattr(gs.constants.backend, backend)
    if backend != gs.cpu:
        yield
        return

    insert, entry, reverse, exit_ = operator.attrgetter("insert", "entry", "reverse", "__exit__")(LaunchRecorder)
    # Per recorder, released with it since a later one may take its identity: the arrays its closed entries used, the
    # values before its open entry, the launches of that entry, and the restores of upstream's rule by entry
    arrays_used, values_before, launches_open, entries_restores = (weakref.WeakKeyDictionary() for _ in range(4))
    mismatches = set()
    # The copies of the checker are kernel launches of their own without zero-copy views, which it does not record
    is_copying = False

    def iter_arrays(kernel, args, is_used):
        # The arrays of the struct arguments of a launch, those its compiled form uses alone with `is_used`, and the
        # arrays it takes directly otherwise
        key = kernel.ensure_compiled(*args)
        flat_names_used = kernel.used_py_dataclass_parameters_by_key_enforcing.get(key) if is_used else None
        for arg_meta, arg in zip(kernel.arg_metas, args):
            if isinstance(arg, (qd.Tensor, qd.Field, qd.Ndarray)) and not is_used:
                yield (arg_meta.name,), arg
            elif dataclasses.is_dataclass(arg):
                stack = [((arg_meta.name,), arg)]
                while stack:
                    path, struct = stack.pop()
                    for field in dataclasses.fields(struct):
                        value = getattr(struct, field.name)
                        if isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
                            array_path = (*path, field.name)
                            if flat_names_used is None or "__qd_" + "__qd_".join(array_path) in flat_names_used:
                                yield array_path, value
                        elif dataclasses.is_dataclass(value):
                            stack.append(((*path, field.name), value))

    def own_view(array):
        # The checker writes the arrays through zero-copy views of its own, None for an array without any, since a
        # write through the view an array caches bumps the version of the tensors the getters handed out, which autograd
        # may have saved
        if qd_zerocopy_view(array) is None:
            return None
        return (array._unwrap() if isinstance(array, qd.Tensor) else array).to_torch(copy=False)

    def write(array, value):
        view = own_view(array)
        if view is None:
            qd_from_torch(array, value)
        else:
            view.copy_(value)

    def is_equal(value, value_ref):
        # Entries the solver leaves at NaN compare equal to themselves
        if value.is_floating_point():
            return torch.allclose(value, value_ref, rtol=0.0, atol=0.0, equal_nan=True)
        return torch.equal(value, value_ref)

    def close_entry(recorder):
        nonlocal is_copying
        values = values_before.pop(recorder, {})
        launches = launches_open.pop(recorder, [])
        if not launches:
            return
        is_copying = True
        restores = [
            (path, array, value)
            for path, array, value in values.values()
            if not is_equal(qd_to_torch(array, copy=True), value)
        ]
        is_copying = False
        entries_restores.setdefault(recorder, {})[len(recorder.entries) - 1] = restores
        recorder_arrays_used = arrays_used.setdefault(recorder, set())
        for kernel, args in launches:
            recorder_arrays_used.update(id(array) for _, array in iter_arrays(kernel, args, is_used=True))

    def checked_insert(recorder, kernel, args):
        nonlocal is_copying
        if is_copying:
            return
        if not recorder._is_grouping:
            close_entry(recorder)
        recorder_arrays_used = arrays_used.get(recorder, set())
        values = values_before.setdefault(recorder, {})
        is_copying = True
        for path, array in iter_arrays(kernel, args, is_used=True):
            if id(array) in recorder_arrays_used and id(array) not in values:
                values[id(array)] = (".".join(path), array, qd_to_torch(array, copy=True))
        is_copying = False
        launches_open.setdefault(recorder, []).append((kernel, args))
        insert(recorder, kernel, args)

    def checked_entry(recorder, *args, **kwargs):
        close_entry(recorder)
        return entry(recorder, *args, **kwargs)

    def checked_reverse(recorder, start=0, stop=None):
        nonlocal is_copying
        close_entry(recorder)
        entries_idx = range(len(recorder.entries))[start:stop]
        entries = recorder.entries[start:stop]
        # Every array the launches reach, whose values and adjoints both reversals start from
        arrays = {}
        for entry_ in entries:
            for kernel, args in entry_.launches:
                for path, array in iter_arrays(kernel, args, is_used=False):
                    arrays.setdefault(id(array), (f"{kernel.func.__name__}:{'.'.join(path)}", array))
        is_copying = True
        values_start = {
            key: (qd_to_torch(array, copy=True), qd_to_torch(array.grad, copy=True) if array.has_grad() else None)
            for key, (_, array) in arrays.items()
        }
        is_copying = False

        restores = entries_restores.get(recorder, {})
        arrays_saved = [list(entry_.arrays_saved) for entry_ in entries]
        for i_e, entry_ in zip(entries_idx, entries):
            entry_.arrays_saved[:] = [(array, own_view(array), value) for _, array, value in restores.get(i_e, ())]
        try:
            tensors_grad_ref = reverse(recorder, start, stop)
        finally:
            for entry_, saved in zip(entries, arrays_saved):
                entry_.arrays_saved[:] = saved
        is_copying = True
        grads_ref = {key: qd_to_torch(array.grad, copy=True) for key, (_, array) in arrays.items() if array.has_grad()}
        for key, (_, array) in arrays.items():
            value, grad = values_start[key]
            write(array, value)
            if grad is not None:
                write(array.grad, grad)
        is_copying = False

        tensors_grad = reverse(recorder, start, stop)
        is_copying = True
        arrays_mismatch = [
            arrays[key][0] for key, grad in grads_ref.items() if not is_equal(qd_to_torch(arrays[key][1].grad), grad)
        ]
        is_copying = False
        arrays_mismatch += [
            "torch input"
            for key, grad in tensors_grad_ref.items()
            if key not in tensors_grad or not is_equal(tensors_grad[key], grad)
        ]
        if arrays_mismatch:
            # The overwrites the recorder left unsaved, where its gradients depart from upstream's rule
            overwrites = [
                f"{recorder.entries[i_e].launches[0][0].func.__name__} overwrites {path}"
                for i_e in entries_idx
                for path, array, _ in restores.get(i_e, ())
                if all([array is not saved for saved, *_ in recorder.entries[i_e].arrays_saved])
            ]
            mismatches.add(f"adjoints {sorted(set(arrays_mismatch))} after unsaved {sorted(set(overwrites))}")
        return tensors_grad

    # A recorder of the calls between two steps is exited after each call, before the step runs unrecorded
    def checked_exit(recorder, *args):
        close_entry(recorder)
        return exit_(recorder, *args)

    monkeypatch.setattr(LaunchRecorder, "insert", checked_insert)
    monkeypatch.setattr(LaunchRecorder, "entry", checked_entry)
    monkeypatch.setattr(LaunchRecorder, "reverse", checked_reverse)
    monkeypatch.setattr(LaunchRecorder, "__exit__", checked_exit)
    yield
    assert not mismatches, f"Gradients of the backward pass differ under upstream's rule: {sorted(mismatches)}"


def _add_hinge_arm(parent, body_name, pos, axis="0 1 0", joint_pos=None, **joint_kwargs):
    """Add a 1-DOF hinge arm link (y-axis hinge by default, capsule geom, explicit inertia) and return its body
    element."""
    body = ET.SubElement(parent, "body", name=body_name, pos=pos)
    if joint_pos is not None:
        joint_kwargs["pos"] = joint_pos
    ET.SubElement(body, "joint", type="hinge", axis=axis, **joint_kwargs)
    ET.SubElement(body, "inertial", mass="0.5", pos="0.1 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
    return body


@pytest.fixture(scope="session")
def grad_free():
    mjcf = ET.Element("mujoco", model="free")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="chassis", pos="0 0 0")
    # The anisotropic inertia, the fixed child moving the center of mass off the joint origin and the armature coupling
    # the translation to the rotation drive every path of the midpoint integration of a free body
    ET.SubElement(body, "joint", type="free", armature="0.05")
    ET.SubElement(body, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.1 0.2 0.3")
    ET.SubElement(body, "geom", type="box", size="0.1 0.1 0.1", contype="0", conaffinity="0")
    payload = ET.SubElement(body, "body", name="payload", pos="0.2 0.1 0")
    ET.SubElement(payload, "inertial", mass="0.5", pos="0 0 0", diaginertia="0.01 0.02 0.03")
    ET.SubElement(payload, "geom", type="sphere", size="0.05", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_revolute():
    mjcf = ET.Element("mujoco", model="revolute")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm", "0 0 0", stiffness="2.0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_fixed_base():
    mjcf = ET.Element("mujoco", model="fixed_base")
    worldbody = ET.SubElement(mjcf, "worldbody")
    # A base without geometry takes a pose of its own in each environment. The arm hangs off the origin of the base, so
    # that its position follows the orientation of the base, and the hinge sits off the origin of the arm, so that the
    # velocity of the arm moves the origin, and with it the gradient the pose of the base receives.
    base = ET.SubElement(worldbody, "body", name="base", pos="0 0 0")
    ET.SubElement(base, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.1 0.1 0.1")
    _add_hinge_arm(base, "arm", "0.2 0.1 0", joint_pos="-0.1 0 0", stiffness="2.0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_revolute_frictionloss():
    mjcf = ET.Element("mujoco", model="revolute_frictionloss")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm", "0 0 0", frictionloss="0.5")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_prismatic():
    mjcf = ET.Element("mujoco", model="prismatic")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="slider", pos="0 0 0")
    ET.SubElement(body, "joint", type="slide", axis="1 0 0")
    ET.SubElement(body, "inertial", mass="0.5", pos="0 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="box", size="0.05 0.05 0.05", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_spherical():
    mjcf = ET.Element("mujoco", model="spherical")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="ball", pos="0 0 0")
    ET.SubElement(body, "joint", type="ball")
    ET.SubElement(body, "inertial", mass="0.5", pos="0.1 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_capsule():
    mjcf = ET.Element("mujoco", model="capsule")
    ET.SubElement(mjcf, "compiler", angle="degree")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="capsule", pos="0 0 0")
    ET.SubElement(body, "geom", type="capsule", size="0.1 0.2")
    ET.SubElement(body, "joint", name="capsule_joint", type="free")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_capsule_ground():
    mjcf = ET.Element("mujoco", model="capsule_ground")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="capsule", fromto="1.3 0 -0.1 1.7 0 -0.1", size="0.1")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_free_with_revolute():
    mjcf = ET.Element("mujoco", model="free_with_child")
    worldbody = ET.SubElement(mjcf, "worldbody")
    chassis = ET.SubElement(worldbody, "body", name="chassis", pos="0 0 0")
    ET.SubElement(chassis, "freejoint")
    ET.SubElement(chassis, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.1 0.1 0.1")
    ET.SubElement(chassis, "geom", type="box", size="0.1 0.1 0.1", contype="0", conaffinity="0")
    arm = ET.SubElement(chassis, "body", name="arm", pos="0.2 0 0")
    ET.SubElement(arm, "joint", type="hinge", axis="0 1 0")
    ET.SubElement(arm, "inertial", mass="0.5", pos="0.1 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(arm, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_free_with_hinge_ball_slide():
    # A free chassis carrying one arm per other joint type, so that every way of deriving dof positions from the
    # configuration is exercised: a coordinate, a coordinate offset by the neutral configuration, and Euler angles.
    mjcf = ET.Element("mujoco", model="free_with_hinge_ball_slide")
    worldbody = ET.SubElement(mjcf, "worldbody")
    chassis = ET.SubElement(worldbody, "body", name="chassis", pos="0 0 0.5")
    ET.SubElement(chassis, "freejoint")
    ET.SubElement(chassis, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.1 0.1 0.1")
    ET.SubElement(chassis, "geom", type="box", size="0.1 0.1 0.1", contype="0", conaffinity="0")
    _add_hinge_arm(chassis, "hinge_arm", "0.1 0 0")
    for name, pos, joint_kwargs in (
        ("ball_arm", "-0.1 0 0", {"type": "ball"}),
        ("slide_arm", "0 0.1 0", {"type": "slide", "axis": "0 1 0"}),
    ):
        arm = ET.SubElement(chassis, "body", name=name, pos=pos)
        ET.SubElement(arm, "joint", **joint_kwargs)
        ET.SubElement(arm, "inertial", mass="0.5", pos="0.1 0 0", diaginertia="0.01 0.01 0.01")
        ET.SubElement(arm, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_chain3():
    # Hinge -> offset slide -> hinge: the middle slide joint carries a position offset so its anchor depends on the
    # moving parent orientation, and the chain mixes joint types within one entity.
    mjcf = ET.Element("mujoco", model="chain3")
    worldbody = ET.SubElement(mjcf, "worldbody")
    parent = worldbody
    for name in ("l1", "l2", "l3"):
        body = ET.SubElement(parent, "body", name=name, pos="0 0 0" if name == "l1" else "0.2 0 0")
        if name == "l2":
            ET.SubElement(body, "joint", type="slide", axis="1 0 0", pos="0.05 0.02 0.03")
        else:
            ET.SubElement(body, "joint", type="hinge", axis="0 1 0")
        ET.SubElement(body, "inertial", mass="0.3", pos="0.1 0 0", diaginertia="0.005 0.005 0.005")
        ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
        parent = body
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_slider_limit():
    mjcf = ET.Element("mujoco", model="slider_limit")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="cart", pos="0 0 0")
    # A limit close to the origin keeps the rounding of the position under a finite difference small, and a soft one
    # keeps the final position sensitive to a force applied at any step of a rollout that ends on it.
    ET.SubElement(
        body, "joint", name="slider", type="slide", axis="1 0 0", range="-0.5 0.5", damping="0.0", solreflimit="0.15 1"
    )
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1.0", diaginertia="1.0 1.0 1.0")
    ET.SubElement(body, "geom", type="box", size="0.25 0.25 0.1", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_hinge_pair_joint_eq_linear():
    mjcf = ET.Element("mujoco", model="hinge_pair_joint_eq_linear")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm1 = _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(arm1, "arm2", "0.2 0 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "joint", joint1="j1", joint2="j2", polycoef="0 1 0 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_hinge_pair_joint_eq_quadratic():
    mjcf = ET.Element("mujoco", model="hinge_pair_joint_eq_quadratic")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm1 = _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(arm1, "arm2", "0.2 0 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "joint", joint1="j1", joint2="j2", polycoef="0 1 0.5 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_hinge_joint_eq_constant():
    mjcf = ET.Element("mujoco", model="hinge_joint_eq_constant")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(worldbody, "arm2", "0.4 0 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(equality, "joint", joint1="j1", polycoef="0.25 1 0 0 0", solimp="0.95 0.99 0.001", solref="0.005 1")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_connect_loop():
    # arm2 hangs off arm1 and the connect closes the loop within one kinematic tree: the constraint rows then share
    # arm1's dof across both chains (dedup path) and both anchors move with the chain (velocity-product bias).
    mjcf = ET.Element("mujoco", model="connect_loop")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm1 = _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(arm1, "arm2", "0.2 0 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "connect", body1="arm2", body2="arm1", anchor="0.2 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_weld_pair():
    # arm2 hangs off arm1 about a skew axis and the weld ties it back to arm1: the constraint rows share arm1's dof
    # across both chains (dedup path), and the nested hinge's angular velocity-product bias (parent angular velocity
    # cross child axis) is nonzero. A weld between parallel-axis or single-hinge chains would leave the rotation
    # rows' velocity bias identically zero and its adjoint untested.
    mjcf = ET.Element("mujoco", model="weld_pair")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm1 = _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(arm1, "arm2", "0.2 0 0", name="j2", axis="1 0 0")
    equality = ET.SubElement(mjcf, "equality")
    # relpose is the pose of body2 in body1's frame at the welded configuration; a wrong sign leaves the weld
    # violated at rest and turns the scenario into a violent snap instead of a hold.
    ET.SubElement(
        equality,
        "weld",
        body1="arm2",
        body2="arm1",
        relpose="-0.2 0 0 1 0 0 0",
        solimp="0.95 0.99 0.001",
        solref="0.005 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_all_eq_fric():
    # Integration scene exercising every differentiated constraint group: frictionloss on j1, equality JOINT between
    # j1 and j2, equality CONNECT between arm3 and arm4, equality WELD between arm6's nested child and arm5. Each
    # group acts on a disjoint pair of links so the constraint solver faces a well-posed system within every pair.
    # The weld chain nests a skew-axis hinge (see grad_weld_pair) so its rotation rows carry a nonzero velocity bias
    # across two separate trees.
    mjcf = ET.Element("mujoco", model="all_eq_fric")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1", frictionloss="0.5")
    for i_arm in range(2, 7):
        arm = _add_hinge_arm(worldbody, f"arm{i_arm}", f"0 {0.2 * (i_arm - 1):.1f} 0", name=f"j{i_arm}")
        if i_arm == 6:
            # Skew axis for a nonzero rotation-row velocity bias; the joint position offset keeps the hinge axis
            # off the welded point so the weld position rows engage (an axis through it leaves the spin resisted
            # only by the weak rotation rows and the chain winds up).
            _add_hinge_arm(arm, "arm6b", "0.2 0 0", name="j6b", axis="1 0 0", joint_pos="0.05 0.02 0.06")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "joint", joint1="j1", joint2="j2", polycoef="0 1 0 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    ET.SubElement(
        equality, "connect", body1="arm3", body2="arm4", anchor="0.2 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    # The softer solref keeps the 6-row weld on this three-dof chain clear of the stabilization stability
    # boundary (the shared 0.005 timeconst is clamped to 2 * substep_dt, which is marginal here).
    ET.SubElement(
        equality,
        "weld",
        body1="arm6b",
        body2="arm5",
        relpose="-0.2 -0.2 0 1 0 0 0",
        solimp="0.95 0.99 0.001",
        solref="0.02 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_cartpole():
    mjcf = ET.Element("mujoco", model="cartpole")
    worldbody = ET.SubElement(mjcf, "worldbody")
    cart = ET.SubElement(worldbody, "body", name="cart", pos="0 0 0")
    # Same limit as grad_slider_limit, for the same reasons.
    ET.SubElement(
        cart, "joint", name="slider", type="slide", axis="1 0 0", range="-0.5 0.5", damping="0.0", solreflimit="0.15 1"
    )
    ET.SubElement(cart, "inertial", pos="0 0 0", mass="1.0", diaginertia="1.0 1.0 1.0")
    ET.SubElement(cart, "geom", type="box", size="0.25 0.25 0.1", contype="0", conaffinity="0", rgba="0 0 0.8 1")
    pole = ET.SubElement(cart, "body", name="pole", pos="0 0 0")
    ET.SubElement(pole, "joint", name="hinge", type="hinge", axis="0 1 0", damping="0.0")
    ET.SubElement(pole, "inertial", pos="0 0 0.5", mass="10.0", diaginertia="1.0 1.0 1.0")
    ET.SubElement(
        pole, "geom", type="box", pos="0 0 0.5", size="0.025 0.025 0.5", contype="0", conaffinity="0", rgba="1 1 1 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_hopper():
    mjcf = ET.Element("mujoco", model="hopper")
    ET.SubElement(mjcf, "compiler", angle="radian")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", limited="true", armature="1", damping="1")
    ET.SubElement(default, "geom", condim="3", friction="0.9 0.005 0.0001")
    worldbody = ET.SubElement(mjcf, "worldbody")
    torso = ET.SubElement(worldbody, "body", name="torso", pos="0 0 1.25")
    root_kwargs = {"pos": "0 0 0", "limited": "false", "armature": "0", "damping": "0"}
    ET.SubElement(torso, "joint", name="rootx", axis="1 0 0", type="slide", **root_kwargs)
    ET.SubElement(torso, "joint", name="rootz", axis="0 0 1", type="slide", **root_kwargs)
    ET.SubElement(torso, "joint", name="rooty", axis="0 1 0", type="hinge", **root_kwargs)
    ET.SubElement(torso, "geom", name="torso_geom", type="capsule", size="0.05 0.2")
    thigh = ET.SubElement(torso, "body", name="thigh", pos="0 0 -0.2")
    ET.SubElement(thigh, "joint", name="thigh_joint", pos="0 0 0", axis="0 -1 0", type="hinge", range="-2.61799 0")
    ET.SubElement(thigh, "geom", name="thigh_geom", type="capsule", size="0.05 0.225", pos="0 0 -0.225")
    leg = ET.SubElement(thigh, "body", name="leg", pos="0 0 -0.7")
    ET.SubElement(leg, "joint", name="leg_joint", pos="0 0 0.25", axis="0 -1 0", type="hinge", range="-2.61799 0")
    ET.SubElement(leg, "geom", name="leg_geom", type="capsule", size="0.04 0.25")
    foot = ET.SubElement(leg, "body", name="foot", pos="0 0 -0.25")
    ET.SubElement(
        foot, "joint", name="foot_joint", pos="0 0 0", axis="0 -1 0", type="hinge", range="-0.785398 0.785398"
    )
    ET.SubElement(
        foot,
        "geom",
        name="foot_geom",
        type="capsule",
        size="0.06 0.195",
        pos="0.06 0 0",
        quat="0.707107 0 -0.707107 0",
        friction="2 0.005 0.0001",
    )
    return ET.tostring(mjcf, encoding="unicode")

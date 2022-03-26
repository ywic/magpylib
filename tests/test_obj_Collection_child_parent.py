import numpy as np
import magpylib as magpy

def test_parent_setter():
    """ setting and removing a parent"""
    child_labels = lambda x: [c.style.label for c in x]

    # default parent is None
    x1 = magpy.Sensor(style_label='x1')
    assert x1.parent is None

    # init collection gives parent
    c1 = magpy.Collection(x1, style_label='c1')
    assert x1.parent.style.label == 'c1'
    assert child_labels(c1) == ['x1']

    # remove parent with setter
    x1.parent=None
    assert x1.parent is None
    assert child_labels(c1) == []

    # set parent
    x1.parent = c1
    assert x1.parent.style.label == 'c1'
    assert child_labels(c1) == ['x1']

    # set another parent
    c2 = magpy.Collection(style_label='c2')
    x1.parent = c2
    assert x1.parent.style.label == 'c2'
    assert child_labels(c1) == []
    assert child_labels(c2) == ['x1']


def test_collection_inputs():
    """ test basic collection inputs"""

    s1 = magpy.magnet.Cuboid(style_label='s1')
    s2 = magpy.magnet.Cuboid(style_label='s2')
    s3 = magpy.magnet.Cuboid(style_label='s3')
    x1 = magpy.Sensor(style_label='x1')
    x2 = magpy.Sensor(style_label='x2')
    c1 = magpy.Collection(x2, style_label='c1')

    c2 = magpy.Collection(c1, x1, s1, s2, s3)
    assert [c.style.label for c in c2.children] == ['c1', 'x1', 's1', 's2', 's3']
    assert [c.style.label for c in c2.sensors] == ['x1']
    assert [c.style.label for c in c2.sources] == ['s1', 's2', 's3']
    assert [c.style.label for c in c2.collections] == ['c1']


def test_collection_parent_child_relation():
    """ test if parent-child relations are properly set with collections"""

    s1 = magpy.magnet.Cuboid()
    s2 = magpy.magnet.Cuboid()
    s3 = magpy.magnet.Cuboid()
    x1 = magpy.Sensor()
    x2 = magpy.Sensor()
    c1 = magpy.Collection(x2)
    c2 = magpy.Collection(c1, x1, s1, s2, s3)

    assert x1.parent == c2
    assert s3.parent == c2
    assert x2.parent == c1
    assert c1.parent == c2
    assert c2.parent is None


def test_collections_add():
    """ test collection construction"""
    child_labels = lambda x: [c.style.label for c in x]

    x1 = magpy.Sensor(style_label='x1')
    x2 = magpy.Sensor(style_label='x2')
    x3 = magpy.Sensor(style_label='x3')
    x6 = magpy.Sensor(style_label='x6')
    x7 = magpy.Sensor(style_label='x7')

    # simple add
    c2 = magpy.Collection(x1, style_label='c2')
    c2.add(x2, x3)
    assert child_labels(c2) == ['x1', 'x2', 'x3']

    # adding another collection
    c3 = magpy.Collection(x6, style_label='c3')
    c2.add(c3)
    assert child_labels(c2) == ['x1', 'x2', 'x3', 'c3']
    assert child_labels(c3) == ['x6']

    # adding to child collection should not change its parent collection
    c3.add(x7)
    assert child_labels(c2) == ['x1', 'x2', 'x3', 'c3']
    assert child_labels(c3) == ['x6', 'x7']

    # add with parent override
    assert x7.parent == c3

    c4 = magpy.Collection(style_label='c4')
    c4.add(x7, override_parent=True)

    assert child_labels(c3) == ['x6']
    assert child_labels(c4) == ['x7']
    assert x7.parent == c4


def test_collection_plus():
    """
    testing collection adding and the += functionality
    """
    child_labels = lambda x: [c.style.label for c in x]

    s1 = magpy.magnet.Cuboid(style_label='s1')
    s2 = magpy.magnet.Cuboid(style_label='s2')
    x1 = magpy.Sensor(style_label='x1')
    x2 = magpy.Sensor(style_label='x2')
    x3 = magpy.Sensor(style_label='x3')
    c1 = magpy.Collection(s1, style_label='c1')

    # practical simple +
    c2 = c1 + s2
    assert child_labels(c2) == ['c1', 's2']

    # useless triple addition consistency
    c3 = x1 + x2 + x3
    assert c3[0][0].style.label == 'x1'
    assert c3[0][1].style.label == 'x2'
    assert c3[1].style.label == 'x3'

    # useless += consistency
    s3 = magpy.magnet.Cuboid(style_label='s3')
    c2 += s3
    assert [c.style.label for c in c2[0]] == ['c1', 's2']
    assert c2[1] == s3


def test_collection_remove():
    """ removing from collections"""
    child_labels = lambda x: [c.style.label for c in x]
    source_labels = lambda x: [c.style.label for c in x.sources]
    sensor_labels = lambda x: [c.style.label for c in x.sensors]

    x1 = magpy.Sensor(style_label='x1')
    x2 = magpy.Sensor(style_label='x2')
    x3 = magpy.Sensor(style_label='x3')
    x4 = magpy.Sensor(style_label='x4')
    x5 = magpy.Sensor(style_label='x5')
    s1 = magpy.misc.Dipole(style_label='s1')
    s2 = magpy.misc.Dipole(style_label='s2')
    s3 = magpy.misc.Dipole(style_label='s3')
    q1 = magpy.misc.CustomSource(style_label='q1')
    c1 = magpy.Collection(x1, x2, x3, x4, x5, style_label='c1')
    c2 = magpy.Collection(s1, s2, s3, style_label='c2')
    c3 = magpy.Collection(q1, c1, c2, style_label='c3')

    assert child_labels(c1) == ['x1', 'x2', 'x3', 'x4', 'x5']
    assert child_labels(c2) == ['s1', 's2', 's3']
    assert child_labels(c3) == ['q1', 'c1', 'c2']

    # remove item from collection
    c1.remove(x5)
    assert child_labels(c1) == ['x1', 'x2', 'x3', 'x4']
    assert [c.style.label for c in c1.sensors] == ['x1', 'x2', 'x3', 'x4']

    # remove 2 items from collection
    c1.remove(x3, x4)
    assert child_labels(c1) == ['x1', 'x2']
    assert sensor_labels(c1) == ['x1', 'x2']

    # remove item from child collection
    c3.remove(s3)
    assert child_labels(c3) == ['q1', 'c1', 'c2']
    assert child_labels(c2) == ['s1', 's2']
    assert source_labels(c2) == ['s1', 's2']

    # remove child collection
    c3.remove(c2)
    assert child_labels(c3) == ['q1', 'c1']
    assert child_labels(c2) == ['s1', 's2']

    # attempt remove non-existant child
    c3.remove(s1, errors='ignore')
    assert child_labels(c3) == ['q1', 'c1']
    assert child_labels(c1) == ['x1', 'x2']

    # attempt remove child in lower level with recursion=False
    c3.remove(x1, errors='ignore', recursive=False)
    assert child_labels(c3) == ['q1', 'c1']
    assert child_labels(c1) == ['x1', 'x2']


def test_collection_nested_getBH():
    """ test if getBH functionality is self-consistent with nesting"""
    s1 = magpy.current.Loop(1, 1)
    s2 = magpy.current.Loop(1, 1)
    s3 = magpy.current.Loop(1, 1)
    s4 = magpy.current.Loop(1, 1)

    obs = [(1,2,3), (-2,-3,1), (2,2,-4), (4,2,-4)]
    coll = s1 + s2 + s3 + s4 # nasty nesting

    B1 = s1.getB(obs)
    B4 = coll.getB(obs)
    np.testing.assert_allclose(4*B1, B4)

    H1 = s1.getH(obs)
    H4 = coll.getH(obs)
    np.testing.assert_allclose(4*H1, H4)
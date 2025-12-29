import Value

def test_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()

    assert a.grad == 1.0
    assert b.grad == 1.0

from syftr.helpers import get_exception_report


def test_get_exception_report():
    generation_exceptions = []

    try:
        eval("x === 3")  # SyntaxError
    except Exception as e:
        generation_exceptions.append(e)

    try:
        "2" + 2  # TypeError
    except Exception as e:
        generation_exceptions.append(e)

    try:
        int("abc")  # ValueError
    except Exception as e:
        generation_exceptions.append(e)

    evaluation_exceptions = []

    try:
        lst = [1, 2, 3]
        print(lst[5])  # IndexError
    except Exception as e:
        evaluation_exceptions.append(e)

    try:
        d = {"a": 1}
        print(d["b"])  # KeyError
    except Exception as e:
        evaluation_exceptions.append(e)

    try:
        None.some_method()  # AttributeError
    except Exception as e:
        evaluation_exceptions.append(e)

    try:
        1 / 0  # ZeroDivisionError
    except Exception as e:
        evaluation_exceptions.append(e)

    exceptions = []
    exceptions.append(
        ExceptionGroup(
            "Exceptions during generation",
            generation_exceptions,
        )
    )
    exceptions.append(
        ExceptionGroup(
            "Exceptions during evaluation",
            evaluation_exceptions,
        )
    )
    exception_group = ExceptionGroup("Trial failed", exceptions)
    exception_message = get_exception_report(exception_group)

    assert "SyntaxError" in exception_message
    assert "TypeError" in exception_message
    assert "ValueError" in exception_message
    assert "IndexError" in exception_message
    assert "KeyError" in exception_message
    assert "AttributeError" in exception_message
    assert "ZeroDivisionError" in exception_message
    assert "Trial failed" in exception_message
    assert "Exceptions during generation" in exception_message
    assert "Exceptions during evaluation" in exception_message

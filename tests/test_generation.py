from generation.dummy import EchoGenerator


def test_echo_generator_appends_context() -> None:
    generator = EchoGenerator()
    response = generator.generate("Hello", ["World"])
    assert "Hello" in response
    assert "World" in response

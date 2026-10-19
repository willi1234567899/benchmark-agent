from behave import given, then, when


@given("a fixture setup")
def step_given_setup(context) -> None:
    context.ready = True


@when("the fixture step runs")
def step_when_runs(context) -> None:
    assert context.ready is True


@then("the fixture succeeds")
def step_then_succeeds(context) -> None:
    assert True

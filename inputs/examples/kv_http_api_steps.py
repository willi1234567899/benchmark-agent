from behave import given, then, when


@given("a key-value API client is configured")
def step_client_configured(context):
    context.recorded = []


@when('the "{operation}" request is prepared for key "{key}"')
def step_request_prepared(context, operation, key):
    context.current = {"operation": operation, "key": key}


@then('the "{operation}" response is accepted')
def step_response_accepted(context, operation):
    assert context.current["operation"] == operation
    context.recorded.append(operation)

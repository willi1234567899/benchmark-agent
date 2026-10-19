Feature: Smoke fixture for evaluation harness
  Scenario: trivial pass
    Given a fixture setup
    When the fixture step runs
    Then the fixture succeeds

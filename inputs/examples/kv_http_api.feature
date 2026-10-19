# Multi-shot example from a DIFFERENT domain (generic key-value HTTP API).
# Intentionally not the target store write/read/delete, to avoid contaminating the target workload.

Feature: Key-value HTTP API smoke benchmark

  Scenario Outline: Exercise one key-value operation
    Given a key-value API client is configured
    When the "<operation>" request is prepared for key "demo"
    Then the "<operation>" response is accepted

    Examples:
      | operation |
      | put       |
      | get       |
      | delete    |

@read @singlestore
Feature: Kestrel read benchmark (GET .../series/{series}/points)
  In order to evaluate read performance characteristics
  As a test engineer
  I want to run read benchmarks against the Kestrel metrics service with adjustable parameters

  Background:
    Given a SUT Kestrel endpoint is configured and reachable
    And the target tenant and store from the SUT config are available

  Scenario Outline: generic read benchmark run
    When I run a generic read benchmark on series "<series>" since "<since>" with limit <limit>, <concurrent_clients> concurrent clients and label filter "<label_filter>"
    Then I store the generic read benchmark result as "reports/read-<id>.json"
    And I store the read benchmark context as "reports/read-context-<id>.json"

    @normal
    Examples:
      | id     | series                | since | limit | concurrent_clients | label_filter |
      | smoke  | bddbench_single_write | PT2H  | 25    | 1                  | host=h1      |
      | load   | bddbench_single_write | PT6H  | 1000  | 4                  | none         |
      | stress | bddbench_single_write | P1D   | 10000 | 8                  | host=h1      |
      | spike  | bddbench_single_write | PT1H  | 100   | 32                 | none         |
      | soak   | bddbench_single_write | PT1H  | 100   | 2                  | none         |
